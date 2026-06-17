import torch
import math
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils import tensorboard

from tqdm import tqdm
from einops import rearrange

from cardiac_diffusion.mri import fftnc, image2kspace
from cardiac_diffusion.utils import normalize, tensor_to_gif, save_image
from cardiac_diffusion.metrics import compute_metrics
from cardiac_diffusion.stop_gradient import ScoreWithIdentityGradWrapper


class Variation(object):
    def __init__(
            self, 
            prior_model, 
            measurements,
            mask,
            output_dir,
            model_path,
            unet_type='SpatioTemporalUnet',
            sensitivities=None, 
            multicoil=True,
            sparse_representation=False,
            optim_steps=200,
            reg_param=1.0,
            tv_regularization=False,
            tv_reg_param=0.1,
            take_slab=False,
            slab_size=11,
            circular_padding=False,
            timestep_factor=0.5,
            accumulation_steps=1,
            time_sampling='random',
            slab_sampling='random',
            consistency_loss='l2',
            regularization_loss='l2',
            lr=1e-3,
            lr_scheduler=False,
            lr_scheduler_factor=0.5,
            lr_scheduler_patience=25,
            write_tensorboard=True,
            measure_time=False,
            measure_ser=False
    ):
        self.prior_model = prior_model
        self.measurements = measurements
        self.mask = mask
        self.output_dir = output_dir
        self.sparse_representation = sparse_representation
        self.write_tensorboard = write_tensorboard

        if multicoil:
            assert sensitivities is not None, "Sensitivities must be provided for multicoil data."
            self.sensitivities = sensitivities
            self.fwd = lambda x: image2kspace(x.unsqueeze(2), self.sensitivities.unsqueeze(1), dim=(3, 4))
        else:
            self.fwd = lambda x: fftnc(x, dim=(2, 3))

        self.optim_steps = optim_steps
        self.reg_param = reg_param
        self.tv_regularization = tv_regularization
        self.tv_reg_param = tv_reg_param
        self.timestep_factor = timestep_factor
        self.accumulation_steps = accumulation_steps
        self.time_sampling = time_sampling
        self.slab_sampling = slab_sampling
        self.take_slab = take_slab
        self.slab_size = slab_size
        self.circular_padding = circular_padding

        self.lr = lr
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_factor = lr_scheduler_factor
        self.lr_scheduler_patience = lr_scheduler_patience

        if model_path is not None:
            self.load(model_path)
        else:
            raise ValueError("Model path not provided.")
        
        self.prior_model = ScoreWithIdentityGradWrapper(self.prior_model)

        if consistency_loss == 'l1':
            self.consistency_loss_fn = nn.L1Loss()
        elif consistency_loss == 'l2':
            self.consistency_loss_fn = nn.MSELoss()
        else:
            raise NotImplementedError("Loss type for consistency not implemented. Default: l2")
        
        if regularization_loss == 'l1':
            regularization_loss_fn = nn.L1Loss()
        elif regularization_loss == 'l2':
            regularization_loss_fn = nn.MSELoss()
        elif regularization_loss == 'l2-magnitude':
            to_mag = lambda x: abs(torch.complex(x[:, 0], x[:, 1]))
            regularization_loss_fn = lambda x, y: nn.MSELoss()(to_mag(x), to_mag(y))
        else:
            raise NotImplementedError("Loss type for regularization not implemented. Default: l2")

        if unet_type == 'MiddleFrameUnet' or unet_type == 'MiddleFrameAsymmetricUnet':
            takeMiddleFrame = lambda x: x[:, :, x.shape[2] // 2, :, :]
            self.regularization_loss_fn = lambda x, y: regularization_loss_fn(takeMiddleFrame(x), y)
            assert self.take_slab, "One is required to take a slab for regularization with MiddleFrameUnet."
            assert self.slab_size == self.prior_model.model.restoration_fn.input_frames, "Slab size must be equal to the number of input frames for MiddleFrameUnet."
        else:
            self.regularization_loss_fn = regularization_loss_fn      
        
        if self.write_tensorboard:
            self.writer = tensorboard.SummaryWriter(log_dir=output_dir)

    def remove_module_prefix(self, state_dict):
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v  # Remove 'module.' prefix
            else:
                new_state_dict[k] = v
        return new_state_dict

    def load(self, load_path):
        print("Loading : ", load_path)
        map_location = torch.device(f'cuda:{torch.cuda.current_device()}') 
        model_data = torch.load(load_path, map_location=map_location, weights_only=True) 

        self.step = model_data['step']
        self.prior_model.load_state_dict(self.remove_module_prefix(model_data['ema']))

    def apply_mask(self, kspace):
        if self.sparse_representation:
            # the mask and the measurements are stored in a sparse representation
            # -> removing lines instead of setting them to zero
            ky_indices = self.mask[:, :, :, :, 1]
            ky_expanded = ky_indices.unsqueeze(2).expand(-1, -1, kspace.shape[2], -1, -1) 
            k_selected = kspace.gather(3, ky_expanded)
            return k_selected
        else:
            return kspace * self.mask

    def data_consistency(self, x):
        '''
        Computes a consistency value of x with the measurements.
        '''
        b, c, f, h, w = x.shape
        assert c == 2, "Input tensor must have 2 channels for complex representation."

        x_cx = torch.complex(x[:, 0], x[:, 1])
        kspace_cx = self.fwd(x_cx)
        masked_kspace_cx = self.apply_mask(kspace_cx)
        masked_kspace = torch.stack((masked_kspace_cx.real, masked_kspace_cx.imag), dim=1)

        return self.consistency_loss_fn(masked_kspace, self.measurements)

    def regularization(self, x):
        '''
        Computes a regularization value using the prior model.
        '''
        b, c, f, h, w, device = *x.shape, x.device
        assert c == 2, "Input tensor must have 2 channels for complex representation."

        max_timestep = int((self.prior_model.model.timesteps-1) * self.timestep_factor)
        if self.time_sampling == 'descending':
            steps = self.optim_steps*self.accumulation_steps
            t = max_timestep * (steps - self.iteration) // steps
            t = torch.full((b,), t, device=device)
        else:
            t = torch.randint(0, max_timestep, (b,), device=device) 
        noise = torch.randn_like(x)

        ground = noise if self.prior_model.model.noise_based else x
        estimation = self.prior_model(x, t, noise)

        regularization = self.regularization_loss_fn(ground, estimation)

        reg_factor = self.reg_param * (self.prior_model.model.extract(self.prior_model.model.sqrt_one_minus_alphas_cumprod, t, x.shape) / self.prior_model.model.extract(self.prior_model.model.sqrt_alphas_cumprod, t, x.shape))

        return reg_factor * regularization
    
    def total_variation_regularization(self, x, axes=2):
        """
        Computes the Total Variation (TV) regularization along a specific axis using finite differences.
        If multiple axes are provided, an anisotropic TV regularization is computed.
        """
        # TV in time dimension
        tv_reg = 0
        tv_reg += torch.mean(torch.abs(x[:, :, :-1, :, :] - x[:, :, 1:, :, :]))  # forward difference
        tv_reg += torch.mean(torch.abs(x[:, :, 1:, :, :] - x[:, :, :-1, :, :]))  # backward difference
        
        return tv_reg
    
    def choose_slab(self, x):

        if self.slab_sampling == 'random':
            q = torch.randint(0, x.shape[2], (1,)).item()
            start, stop = q - self.slab_size // 2, q + self.slab_size // 2 + self.slab_size % 2

            if self.slab_size > x.shape[2]:
                self.slab_size = x.shape[2]
            if start < 0:
                start, stop = 0, self.slab_size    
            if stop > x.shape[2]:
                start, stop = x.shape[2] - self.slab_size, x.shape[2]

        elif self.slab_sampling == 'sliding_window':
            if self.slab_size >= x.shape[2]:
                start, stop = 0, x.shape[2]
            else:
                max_slabs = int(math.ceil(x.shape[2] / self.slab_size))
                s = self.iteration % max_slabs
                start, stop = s * self.slab_size, (s + 1) * self.slab_size
            if stop > x.shape[2]:
                start, stop = x.shape[2] - self.slab_size, x.shape[2]

        else:
            raise NotImplementedError("Slab sampling method not implemented. Choose 'random' or 'sliding_window'.")

        # print(f"start: {start}, stop: {stop}, slab_size: {self.slab_size}, f: {x.shape[2]}")
        return x[:, :, start:stop]

    def objective(self, x):
        '''
        Combines the data consistency and regularization terms.
        '''
        if self.take_slab:
            x_reg = self.choose_slab(x)
        else:
            x_reg = x
        
        if self.tv_regularization:
            return self.data_consistency(x) + self.regularization(x_reg) + self.tv_reg_param * self.total_variation_regularization(x)
        
        return self.data_consistency(x) + self.regularization(x_reg)
    
    def fit(self, x):
        '''
        Does optimization to fit the reconstruction using the objective function.
        '''
        optimizer = Adam(
            params=[x], 
            lr=self.lr,
            betas=(0.9, 0.99)
        )

        if self.lr_scheduler:
            scheduler = ReduceLROnPlateau(
                optimizer, 
                factor=self.lr_scheduler_factor, 
                patience=self.lr_scheduler_patience
            )
        else:
            scheduler = None

        for i in tqdm(range(self.optim_steps)):
            optimizer.zero_grad()

            acc_loss = 0
            for j in range(self.accumulation_steps):
                self.iteration = i*self.accumulation_steps + j
                loss = self.objective(x)
                loss.backward()
                acc_loss += loss
            loss = acc_loss / self.accumulation_steps

            if self.lr_scheduler and (i > self.optim_steps*0.05): scheduler.step(loss)
            optimizer.step()

            if self.write_tensorboard:
                self.writer.add_scalar('Loss', loss.item(), i)
                if self.lr_scheduler: self.writer.add_scalar('LR', scheduler.get_last_lr()[0], i)

                if i % (self.optim_steps // 10) == 0:
                    mag_recon = abs(torch.complex(x[:, 0], x[:, 1])).unsqueeze(1)
                    mag_recon = rearrange(mag_recon, 'b c f h w -> (b f) c h w')
                    mag_recon = normalize(mag_recon)
                    self.writer.add_images('Reconstruction/process', mag_recon, i)

        if self.write_tensorboard:
            mag_recon = abs(torch.complex(x[:, 0], x[:, 1])).unsqueeze(1)
            mag_recon = rearrange(mag_recon, 'b c f h w -> (b f) c h w')
            mag_recon = normalize(mag_recon)
            self.writer.add_images('Reconstruction/process', mag_recon, self.optim_steps)

        self.save_results(x)
    
    def save_results(self, reconstruction):
        '''
        Saves the reconstruction as a tensor and a GIF.
        '''
        b, c, f, h, w = reconstruction.shape
        reconstruction = reconstruction.detach()

        torch.save(reconstruction, self.output_dir / 'reconstruction.pt')

        if c == 1:
            mag_recon = (reconstruction + 1) * 0.5
        else:
            assert c == 2, "Complex images are only supported for two-channel images."
            mag_recon = abs(torch.complex(reconstruction[:, 0], reconstruction[:, 1])).unsqueeze(1)

        assert b == 1, "Batch size must be 1 for visualization. Other not implemented yet."

        # overwrite the file with complex valued tensor [f, h, w]
        torch.save(torch.complex(reconstruction[:, 0], reconstruction[:, 1]).squeeze(0), self.output_dir / 'reconstruction.pt')

        mag_recon = rearrange(mag_recon, 'b c f h w -> (b f) c h w')
        tensor_to_gif(mag_recon, self.output_dir / 'reconstruction.gif', duration=1.)

        # # crop the image to the center 128x128
        # _, _, h, w = mag_recon.shape
        # mag_recon = mag_recon[:, :, h//2-64:h//2+64, w//2-64:w//2+64]

        if self.write_tensorboard:
            tensor_to_gif(mag_recon, '/florian/masked_k-space_diffusion/result_media/inference_recon.gif', duration=1.)
    
    def evaluate(self, reconstruction, ground_truth):
        '''
        Computes metrics and visualizes the reconstruction and ground truth.
        '''
        reconstruction = reconstruction.detach()

        if ground_truth is None:
            return

        mag_ground = abs(ground_truth).unsqueeze(1)
        mag_ground = rearrange(mag_ground, 'b c f h w -> (b f) c h w')

        mag_recon = abs(torch.complex(reconstruction[:, 0], reconstruction[:, 1])).unsqueeze(1)
        mag_recon = rearrange(mag_recon, 'b c f h w -> (b f) c h w')
        ssims, psnrs, nmses, metric_recon = compute_metrics(mag_ground, mag_recon)

        if self.write_tensorboard:
            mag_ground = normalize(mag_ground, batch_independent=True)
            tensor_to_gif(mag_ground, '/florian/masked_k-space_diffusion/result_media/inference_ground.gif', duration=1.)
            save_image(mag_ground[0], '/florian/masked_k-space_diffusion/result_media/inference_ground.png')
            save_image(metric_recon[0], '/florian/masked_k-space_diffusion/result_media/inference_recon.png')
            
            self.writer.add_images('Reconstruction/ground', mag_ground, 0)        
            self.writer.add_images('Reconstruction/evaluation', metric_recon, 0)
            self.writer.flush()

        return ssims, psnrs, nmses
