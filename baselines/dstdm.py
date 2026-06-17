import copy
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from einops import rearrange
from torch.utils import data, tensorboard

from cardiac_diffusion.dataset import CMRxProcessed
from cardiac_diffusion.mri import image2kspace, kspace2image
from cardiac_diffusion.trainer import EMA
from cardiac_diffusion.unet import UNetModel


def _default_unet_config():
    return {
        "dims": 2,
        "channel_mult": (1, 2, 4),
        "in_channels": 2,
        "out_channels": 2,
        "model_channels": 128,
        "num_res_blocks": 2,
        "attention_resolutions": [],
        "num_heads": 4,
        "dropout": 0.0,
        "use_scale_shift_norm": True,
        "resblock_updown": True,
    }


def _load_checkpoint_payload(path, device="cpu"):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def _extract_checkpoint_state_dict(payload):
    if not isinstance(payload, dict):
        return payload
    return payload.get("ema", payload.get("model", payload))


def _get_unet_config_from_payload(payload):
    if not isinstance(payload, dict):
        return None
    for key in ("UNetModel", "unet", "unet_config"):
        if key in payload and isinstance(payload[key], dict):
            return dict(payload[key])
    return None


class DiffusionDSTDM(nn.Module):
    def __init__(self, restoration_fn, timesteps=100, sigma_min=0.01, sigma_max=50.0, ema_decay=0.999):
        super().__init__()
        self.restoration_fn = restoration_fn
        self.timesteps = timesteps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        sigma_steps = max(1, timesteps - 1)
        sigmas = torch.tensor(
            [sigma_min * (sigma_max / sigma_min) ** (k / sigma_steps) for k in range(timesteps)],
            dtype=torch.float32,
        )
        self.register_buffer("sigmas", sigmas)

        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.restoration_fn)

    def sigma_at(self, step, x_shape):
        step = torch.as_tensor(step, device=self.sigmas.device, dtype=torch.long)
        if step.ndim == 0:
            step = step.unsqueeze(0)
        step = step.clamp(0, self.timesteps - 1)
        sigma = self.sigmas[step]
        return sigma.reshape(step.shape[0], *((1,) * (len(x_shape) - 1)))

    def training_loss(self, x0):
        batch_size = x0.shape[0]
        steps = torch.randint(0, self.timesteps, (batch_size,), device=x0.device)
        sigma = self.sigma_at(steps, x0.shape)
        noise = torch.randn_like(x0)
        xk = x0 + sigma * noise
        prediction = self.restoration_fn(xk, steps)
        return (sigma * prediction + (xk - x0) / sigma).pow(2).mean()

    @torch.no_grad()
    def step(self, xk, step_index, use_ema=True):
        step = torch.full((xk.shape[0],), step_index, device=xk.device, dtype=torch.long)
        sigma_k = self.sigma_at(step, xk.shape)
        sigma_prev_value = self.sigmas[step_index - 1] if step_index > 0 else torch.zeros((), device=xk.device, dtype=xk.dtype)
        sigma_prev = sigma_prev_value.reshape(1, *((1,) * (xk.ndim - 1)))

        model = self.ema_model if use_ema else self.restoration_fn
        score = model(xk, step)

        step_variance = torch.clamp(sigma_k.pow(2) - sigma_prev.pow(2), min=0.0)
        noise = torch.randn_like(xk) if step_index > 0 else torch.zeros_like(xk)
        return xk + step_variance * score + torch.sqrt(step_variance) * noise

    @torch.no_grad()
    def sample(self, x, use_ema=True):
        current = x
        for step_index in reversed(range(self.timesteps)):
            current = self.step(current, step_index, use_ema=use_ema)
        return current

    @staticmethod
    def _prepare_ground_batch(batch, device):
        if isinstance(batch, (tuple, list)):
            ground = batch[0]
        else:
            ground = batch

        ground = ground.to(device)
        if torch.is_complex(ground):
            if ground.ndim == 3:
                ground = ground.unsqueeze(0)
            ground = torch.stack((ground.real, ground.imag), dim=1)
        elif ground.ndim == 4 and ground.shape[0] == 2:
            ground = ground.unsqueeze(0)
        elif ground.ndim == 5 and ground.shape[1] == 2:
            pass
        else:
            raise ValueError(f"Unsupported ground-truth shape for dSTDM training: {tuple(ground.shape)}")

        return ground

    @staticmethod
    def _slice_magnitude(x):
        return torch.sqrt(x[:, 0].pow(2) + x[:, 1].pow(2) + 1e-12)
    
    def _compute_random_val_loss_step(self, val_dataloader, device, slice_batch_size, drop_last_slice_batch):
            dataset = getattr(val_dataloader, "dataset", None)
            val_batch = None

            if dataset is not None and len(dataset) > 0:
                idx = torch.randint(0, len(dataset), (1,)).item()
                sample = dataset[idx]
                if torch.is_tensor(sample):
                    val_batch = sample.unsqueeze(0)
                elif isinstance(sample, (tuple, list)):
                    collated = []
                    for item in sample:
                        if torch.is_tensor(item):
                            collated.append(item.unsqueeze(0))
                        else:
                            collated.append(item)
                    val_batch = tuple(collated)

            if val_batch is None:
                try:
                    val_batch = next(iter(val_dataloader))
                except StopIteration:
                    return None

            val_ground = self._prepare_ground_batch(val_batch, device)
            val_slices, _ = dSTDM.extract_slice_static(val_ground, column=False)
            if val_slices.shape[0] == 0:
                return None

            if val_slices.shape[0] > slice_batch_size:
                rand_ids = torch.randperm(val_slices.shape[0], device=val_slices.device)[:slice_batch_size]
                val_slice_batch = val_slices[rand_ids]
            else:
                val_slice_batch = val_slices

            if drop_last_slice_batch and val_slice_batch.shape[0] < slice_batch_size:
                return None

            return self.training_loss(val_slice_batch).item()

    def fit(
        self,
        dataloader,
        val_dataloader=None,
        lr=2e-4,
        num_epochs=100,
        save_every=10,
        ema_decay=0.999,
        device=None,
        model_dir=None,
        log_dir=None,
        slice_batch_size=64,
        drop_last_slice_batch=False,
        val_max_batches=64,
        checkpoint_metadata=None,
    ):
        if device is None:
            device = next(self.parameters()).device

        writer = tensorboard.SummaryWriter(log_dir=log_dir) if log_dir is not None else None

        optimizer = torch.optim.Adam(self.restoration_fn.parameters(), lr=lr)
        self.ema = EMA(ema_decay)
        self.ema_model.load_state_dict(self.restoration_fn.state_dict())

        self.train()
        global_step = 0
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            for batch in dataloader:
                ground = self._prepare_ground_batch(batch, device)

                # Training follows the paper setup on ny-nt slices only.
                slices, _ = dSTDM.extract_slice_static(ground, column=False)

                total_slices = slices.shape[0]
                for start in range(0, total_slices, slice_batch_size):
                    end = min(start + slice_batch_size, total_slices)
                    if drop_last_slice_batch and (end - start) < slice_batch_size:
                        continue

                    slice_batch = slices[start:end]
                    loss = self.training_loss(slice_batch)

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    self.ema.update_model_average(self.ema_model, self.restoration_fn)

                    epoch_loss += loss.item()
                    num_batches += 1
                    global_step += 1

                    if writer is not None:
                        writer.add_scalar("train/loss_step", loss.item(), global_step)

                    if val_dataloader is not None and writer is not None and global_step % 50 == 0:
                        self.eval()
                        with torch.no_grad():
                            val_loss_step = self._compute_random_val_loss_step(
                                val_dataloader=val_dataloader,
                                device=device,
                                slice_batch_size=slice_batch_size,
                                drop_last_slice_batch=drop_last_slice_batch,
                            )
                        if val_loss_step is not None:
                            writer.add_scalar("val/loss_step", val_loss_step, global_step)
                        self.train()

            avg_epoch_loss = epoch_loss / max(1, num_batches)
            print(f"Epoch {epoch + 1}/{num_epochs} - dSTDM loss: {avg_epoch_loss:.6f}")

            if writer is not None:
                writer.add_scalar("train/loss_epoch", avg_epoch_loss, epoch)

            if val_dataloader is not None:
                self.eval()
                val_epoch_loss = 0.0
                val_num_batches = 0
                vis_logged = False

                with torch.no_grad():
                    for val_batch_idx, val_batch in enumerate(val_dataloader):
                        if val_batch_idx >= val_max_batches:
                            break

                        val_ground = self._prepare_ground_batch(val_batch, device)
                        val_slices, _ = dSTDM.extract_slice_static(val_ground, column=False)

                        total_val_slices = val_slices.shape[0]
                        for start in range(0, total_val_slices, slice_batch_size):
                            end = min(start + slice_batch_size, total_val_slices)
                            if drop_last_slice_batch and (end - start) < slice_batch_size:
                                continue

                            val_slice_batch = val_slices[start:end]
                            val_loss = self.training_loss(val_slice_batch)
                            val_epoch_loss += val_loss.item()
                            val_num_batches += 1

                        if (not vis_logged) and writer is not None and total_val_slices > 0:
                            x0_vis = val_slices[:1]
                            vis_step = torch.full((1,), self.timesteps // 2, dtype=torch.long, device=device)
                            vis_sigma = self.sigma_at(vis_step, x0_vis.shape)
                            xk_vis = x0_vis + vis_sigma * torch.randn_like(x0_vis)
                            pred_vis = self.restoration_fn(xk_vis, vis_step)
                            xhat_vis = xk_vis + vis_sigma.pow(2) * pred_vis
                            xseed_vis = torch.randn_like(x0_vis) * self.sigmas[-1]
                            xsample_vis = self.sample(xseed_vis, use_ema=True)

                            noisy_mag = self._slice_magnitude(xk_vis)
                            output_mag = self._slice_magnitude(xhat_vis)
                            target_mag = self._slice_magnitude(x0_vis)
                            sample_mag = self._slice_magnitude(xsample_vis)

                            noisy_mag = noisy_mag / (noisy_mag.max() + 1e-8)
                            output_mag = output_mag / (output_mag.max() + 1e-8)
                            target_mag = target_mag / (target_mag.max() + 1e-8)
                            sample_mag = sample_mag / (sample_mag.max() + 1e-8)

                            writer.add_image("val/noisy_input", noisy_mag[0].unsqueeze(0), epoch)
                            writer.add_image("val/model_output", output_mag[0].unsqueeze(0), epoch)
                            writer.add_image("val/target", target_mag[0].unsqueeze(0), epoch)
                            writer.add_image("val/sample", sample_mag[0].unsqueeze(0), epoch)
                            vis_logged = True

                avg_val_loss = val_epoch_loss / max(1, val_num_batches)
                print(f"Epoch {epoch + 1}/{num_epochs} - dSTDM val loss: {avg_val_loss:.6f}")
                if writer is not None:
                    writer.add_scalar("val/loss_epoch", avg_val_loss, epoch)

                self.train()

            if model_dir is not None and epoch % save_every == 0 or epoch == num_epochs:
                checkpoint_path = Path(model_dir) / f"epoch={epoch}-step={global_step}.pt"
                self.save(checkpoint_path, step=global_step, checkpoint_metadata=checkpoint_metadata)

        if model_dir is not None:
            final_path = Path(model_dir) / "model.pt"
            self.save(final_path, step=global_step, checkpoint_metadata=checkpoint_metadata)

        if writer is not None:
            writer.close()

    def save(self, path, step=0, checkpoint_metadata=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "step": step,
            "model": self.restoration_fn.state_dict(),
            "ema": self.ema_model.state_dict(),
        }
        if isinstance(checkpoint_metadata, dict):
            payload.update(checkpoint_metadata)
        torch.save(payload, path)


class dSTDM(nn.Module):
    def __init__(self, timesteps=100, sigma_min=0.01, sigma_max=50.0, unet_config=None, ema_decay=0.999):
        super().__init__()
        config = _default_unet_config()
        if unet_config is not None:
            config.update(unet_config)

        self.unet_config = dict(config)

        self.model = UNetModel(**config)
        self.diffusion = DiffusionDSTDM(
            restoration_fn=self.model,
            timesteps=timesteps,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            ema_decay=ema_decay,
        )

    @staticmethod
    def extract_slice_static(x, column=False):
        b, c, f, h, w = x.shape
        if column:
            return rearrange(x, "b c f h w -> (b w) c h f"), w
        return rearrange(x, "b c f h w -> (b h) c w f"), h

    def extract_slice(self, x, column=False):
        return self.extract_slice_static(x, column=column)

    def combine_slices(self, x, s, column=False):
        if column:
            return rearrange(x, "(b w) c h f -> b c f h w", w=s)
        return rearrange(x, "(b h) c w f -> b c f h w", h=s)

    def combine_directions(self, row_image, column_image, lambda_weight=0.5):
        return lambda_weight * row_image + (1.0 - lambda_weight) * column_image

    def fit(
        self,
        dataloader,
        val_dataloader=None,
        lr=2e-4,
        num_epochs=100,
        save_every=10,
        ema_decay=0.999,
        device=None,
        model_dir=None,
        log_dir=None,
        slice_batch_size=64,
        drop_last_slice_batch=False,
        val_max_batches=64,
        checkpoint_metadata=None,
    ):
        return self.diffusion.fit(
            dataloader=dataloader,
            val_dataloader=val_dataloader,
            lr=lr,
            num_epochs=num_epochs,
            save_every=save_every,
            ema_decay=ema_decay,
            device=device,
            model_dir=model_dir,
            log_dir=log_dir,
            slice_batch_size=slice_batch_size,
            drop_last_slice_batch=drop_last_slice_batch,
            val_max_batches=val_max_batches,
            checkpoint_metadata=checkpoint_metadata,
        )

    def forward(self, x, t):
        return self.diffusion.ema_model(x, t)


class ReconstructionDSTDM:
    def __init__(self, args, masked_kspace: torch.Tensor, mask: torch.Tensor, sensitivities: torch.Tensor):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.masked_kspace = masked_kspace.to(self.device)
        self.mask = mask.to(self.device)
        self.sensitivities = sensitivities.to(self.device)

        self.args = self._load_inference_config(args)
        self.training_config = self._load_training_config()
        self.model_path = self._resolve_model_path()
        self.model = self._build_model().to(self.device)
        self.lambda_weight = self._get_model_param("lambda", 0.5)
        self.rho = self._get_model_param("rho", 1.0)

        self.load_model(self.model_path)

    @staticmethod
    def load_yaml(file_path):
        with open(file_path, "r") as file:
            return yaml.safe_load(file)

    def _load_inference_config(self, args):
        inference_config = getattr(args, "inference_config", None)
        if inference_config and Path(inference_config).exists():
            config = self.load_yaml(inference_config)
            for key, value in config.items():
                setattr(args, key, value)
        return args

    def _get_model_param(self, key, default=None):
        model_cfg = getattr(self.args, "model", {}) or {}
        if key in model_cfg:
            return model_cfg.get(key)
        training_model_cfg = (self.training_config or {}).get("model", {})
        return training_model_cfg.get(key, default)

    def _load_training_config(self):
        model_cfg = getattr(self.args, "model", {}) or {}
        model_root = model_cfg.get("path")
        if model_root is None:
            return {}

        config_root = Path(model_root) / "trained_model" / "configs"
        for name in ("dstdm-training.yaml", "diffusion.yaml"):
            path = config_root / name
            if path.exists():
                return self.load_yaml(path)
        return {}

    def _infer_unet_config_from_checkpoint(self):
        if not self.model_path.exists():
            return None

        payload = _load_checkpoint_payload(self.model_path, device="cpu")
        payload_cfg = _get_unet_config_from_payload(payload)
        if payload_cfg is not None:
            return payload_cfg

        state_dict = _extract_checkpoint_state_dict(payload)
        state_dict = self.remove_module_prefix(state_dict)
        default_cfg = _default_unet_config()

        conv_key = "input_blocks.0.0.weight"
        if conv_key in state_dict:
            weight = state_dict[conv_key]
            default_cfg["model_channels"] = int(weight.shape[0])
            default_cfg["in_channels"] = int(weight.shape[1])

        for channel_mult in (
            (1, 2, 2),
            (1, 2, 4),
            (1, 2, 2, 2),
            (1, 1, 2, 2),
            (1, 2, 3),
        ):
            candidate = dict(default_cfg)
            candidate["channel_mult"] = channel_mult
            try:
                probe = UNetModel(**candidate)
                probe.load_state_dict(state_dict, strict=True)
                return candidate
            except RuntimeError:
                continue

        return None

    def _build_model(self):
        timesteps = self._get_model_param("timesteps", 100)
        sigma_min = self._get_model_param("sigma_min", 0.01)
        sigma_max = self._get_model_param("sigma_max", 50.0)
        ema_decay = self._get_model_param("ema_decay", 0.999)

        unet_config = getattr(self.args, "UNetModel", None)
        if unet_config is None and isinstance(self.training_config, dict):
            unet_config = self.training_config.get("UNetModel", self.training_config.get("unet", None))
        if unet_config is None:
            unet_config = self._infer_unet_config_from_checkpoint()
        if unet_config is None:
            unet_config = _default_unet_config()

        return dSTDM(
            timesteps=timesteps,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            unet_config=unet_config,
            ema_decay=ema_decay,
        )

    def _resolve_model_path(self):
        model_cfg = getattr(self.args, "model", {}) or {}
        model_root = model_cfg.get("path")
        checkpoint = model_cfg.get("checkpoint", "model.pt")
        if model_root is None:
            raise ValueError("dSTDM reconstruction requires args.model['path'].")
        return Path(model_root) / "trained_model" / "models" / checkpoint

    def remove_module_prefix(self, state_dict):
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("module."):
                new_state_dict[key[7:]] = value
            else:
                new_state_dict[key] = value
        return new_state_dict

    def load_model(self, load_path):
        model_data = _load_checkpoint_payload(load_path, device=self.device)
        state_dict = model_data.get("ema", model_data.get("model", model_data))
        self.model.diffusion.ema_model.load_state_dict(self.remove_module_prefix(state_dict))
        self.model.eval()
        self.model.diffusion.ema_model.eval()

    def forward_operator(self, image):
        image_complex = torch.complex(image[:, 0], image[:, 1])
        kspace = image2kspace(
            image_complex.unsqueeze(2),
            self.sensitivities.unsqueeze(0).unsqueeze(1),
            dim=(3, 4),
        )
        return kspace * self.mask.unsqueeze(0)

    def adjoint_operator(self, residual):
        image = kspace2image(
            residual,
            sens_maps=self.sensitivities.unsqueeze(0).unsqueeze(1),
            fdim=(3, 4),
            cdim=2,
        )
        return torch.stack((image.real, image.imag), dim=1)

    @torch.no_grad()
    def reverse_step(self, current_image, step_index):
        row_slices, row_size = self.model.extract_slice(current_image, column=False)
        col_slices, col_size = self.model.extract_slice(current_image, column=True)

        row_slices = self.model.diffusion.step(row_slices, step_index, use_ema=True)
        col_slices = self.model.diffusion.step(col_slices, step_index, use_ema=True)

        row_image = self.model.combine_slices(row_slices, row_size, column=False)
        col_image = self.model.combine_slices(col_slices, col_size, column=True)
        combined = self.model.combine_directions(row_image, col_image, lambda_weight=self.lambda_weight)

        residual = self.masked_kspace.unsqueeze(0) - self.forward_operator(combined)
        return combined + self.rho * self.adjoint_operator(residual)

    def reconstruct(self):
        _, frames, _, height, width = self.masked_kspace.unsqueeze(0).shape
        current = torch.randn((1, 2, frames, height, width), device=self.device) * self.model.diffusion.sigmas[-1]

        for step_index in reversed(range(self.model.diffusion.timesteps)):
            current = self.reverse_step(current, step_index)

        recon = torch.view_as_complex(current.squeeze(0).transpose(0, 3).contiguous()).permute(1, 2, 0)
        return recon


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--base", type=Path, required=True, metavar="/path/to/base_directory", help='Path to the base_directory.')
    parser.add_argument(
        "--train_dataset",
        default=[
            "datasets/CineProcessed/2023/TrainingSet",
            "datasets/CineProcessed/2024",
        ],
        nargs="+",
        type=str,
    )
    parser.add_argument(
        "--val_dataset",
        default=[
            "datasets/CineProcessed/2023/ValidationSet",
        ],
        nargs="+",
        type=str,
    )
    parser.add_argument("--output_path", default="models/dSTDM_192x192", type=str)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--save_every", default=10, type=int)
    args = parser.parse_args()

    # Merge data directories with base path
    args.train_dataset = [args.base / p for p in args.train_dataset]
    args.val_dataset = [args.base / p for p in args.val_dataset]
    args.output_path = args.base / args.output_path

    print(f'CUDA: {torch.cuda.is_available(), torch.cuda.device_count()}')

    # Paper defaults / implementation details
    timesteps = 100
    sigma_min = 0.01
    sigma_max = 50.0
    ema_decay = 0.999
    learning_rate = 2e-4
    dataloader_batch_size = 1
    dataloader_num_workers = 8
    slice_batch_size = 192 # 64
    drop_last_slice_batch = False
    lambda_weight = 0.5
    rho = 1.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    unet_config = _default_unet_config()

    model = dSTDM(
        timesteps=timesteps,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        unet_config=unet_config,
        ema_decay=ema_decay,
    ).to(device)

    # Diffusion training data: image_only volumes (no sampling mask generation).
    train_set = CMRxProcessed(
        data_path=args.train_dataset,
        filter=None,
        image_only=True,
        crop_size=(192, 192)
    )

    # Validation data: same setup as training for diffusion prior fitting.
    val_set = CMRxProcessed(
        data_path=args.val_dataset,
        filter=None,
        multicoil=True,
        image_only=True,
        crop_size=(192, 192)
    )

    train_loader = data.DataLoader(
        train_set,
        batch_size=dataloader_batch_size,
        shuffle=True,
        num_workers=dataloader_num_workers,
        drop_last=False,
        pin_memory=True,
    )

    val_loader = data.DataLoader(
        val_set,
        batch_size=dataloader_batch_size,
        shuffle=False,
        num_workers=dataloader_num_workers,
        drop_last=False,
        pin_memory=True,
    )

    out_root = Path(args.output_path)
    model_dir = out_root / "trained_model" / "models"
    config_dir = out_root / "trained_model" / "configs"
    log_dir = out_root / "trained_model" / "logs"
    model_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    config_to_save = {
        "model": {
            "baseline": "dSTDM",
            "path": str(out_root),
            "checkpoint": "model.pt",
            "timesteps": timesteps,
            "sigma_min": sigma_min,
            "sigma_max": sigma_max,
            "ema_decay": ema_decay,
            "lambda": lambda_weight,
            "rho": rho,
        },
        "UNetModel": model.unet_config,
    }
    with open(config_dir / "dstdm-training.yaml", "w") as file:
        yaml.safe_dump(config_to_save, file)

    model.fit(
        dataloader=train_loader,
        val_dataloader=val_loader,
        lr=learning_rate,
        num_epochs=args.epochs,
        save_every=args.save_every,
        ema_decay=ema_decay,
        device=device,
        model_dir=model_dir,
        log_dir=log_dir,
        slice_batch_size=slice_batch_size,
        drop_last_slice_batch=drop_last_slice_batch,
        checkpoint_metadata={"UNetModel": model.unet_config, "model": config_to_save["model"]},
    )