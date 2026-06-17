import torch
import yaml
from pathlib import Path

from cardiac_diffusion.diffusion import Diffusion
from cardiac_diffusion.unet import SpatioTemporalUNetModel, UNetModel

from cardiac_diffusion.mri import image2kspace


class ReconstructionDPS:
    def __init__(self, args, masked_kspace: torch.Tensor, mask: torch.Tensor, sensitivities: torch.Tensor):
        """
        Initialize the reconstruction model.

        Args:
            args: Configuration arguments.
            masked_kspace: Input k-space data.
            mask: Undersampling mask.
            sensitivities: Coil sensitivity maps.
        """
        self.args = args
        self.masked_kspace = torch.stack((masked_kspace.real, masked_kspace.imag), dim=0).unsqueeze(0)
        self.mask = mask.unsqueeze(0)
        self.sensitivities = sensitivities.unsqueeze(0)

        # Load configs from yaml files and add additional parameters to args
        diffusion_config_path = Path(args.model['path']) / 'trained_model' / 'configs' / 'diffusion.yaml'
        diffusion_config = self.load_yaml(diffusion_config_path)
        for key, value in diffusion_config.items():
            setattr(args, key, value)

        # Set output directories
        args.model_path = Path(args.model['path']) / 'trained_model' / 'models' / args.model['checkpoint']
        if args.output_path:
            args.output_dir = args.output_path
        else:
            args.output_dir = Path(args.model['path']) / 'inference' / args.model['output_name']
        args.output_dir.mkdir(exist_ok=True, parents=True)

        # Build the restoration function
        if args.model['unet_type'] == 'SpatioTemporalUNetModel':
            self.restoration_fn = SpatioTemporalUNetModel(
                **args.SpatioTemporalUNetModel
            ).cuda()
        elif args.model['unet_type'] == 'UNetModel':
            self.restoration_fn = UNetModel(
                **args.UNetModel
            ).cuda()
        else:
            raise ValueError(f"Unknown unet_type: {args.model['unet_type']}")

        # Build the diffusion model
        self.diffusion = Diffusion(
            restoration_fn=self.restoration_fn,
            **args.diffusion
        ).cuda()

        # load the trained model
        self.load_model(args.model_path)

    @staticmethod
    def load_yaml(file_path):
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file)
        return config

    @staticmethod
    def save_yaml(config, file_path):
        with open(file_path, 'w') as file:
            yaml.safe_dump(config, file)

    def load_model(self, load_path):
        print("Loading : ", load_path)
        map_location = torch.device(f'cuda:{torch.cuda.current_device()}') 
        model_data = torch.load(load_path, map_location=map_location, weights_only=True) 

        self.step = model_data['step']
        self.diffusion.load_state_dict(self.remove_module_prefix(model_data['ema']))

    def remove_module_prefix(self, state_dict):
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v  # Remove 'module.' prefix
            else:
                new_state_dict[k] = v
        return new_state_dict

    def fwd(self, x):
        x_cx = torch.complex(x[:, 0], x[:, 1])
        kspace_cx = image2kspace(x_cx.unsqueeze(2), self.sensitivities.unsqueeze(1), dim=(3, 4))
        masked_kspace_cx = kspace_cx * self.mask

        return torch.stack((masked_kspace_cx.real, masked_kspace_cx.imag), dim=1)
    
    def DPS_sample(self, x, t, eta=0.95, scale = 20.0):
        """
        DPS reconstruction is DDPM sampling with data consistency guidance.
        """
        x_current = x
        x_current = x_current.requires_grad_(True)

        while t:
            # DPS with gradients
            # x_current = x_current.clone().detach().requires_grad_(True)
            # estimation = self.diffusion.restoration_fn(x_current, t)

            # DPS version without gradients of the network
            with torch.no_grad():
                estimation = self.diffusion.restoration_fn(x_current, t)

            if self.diffusion.noise_based:
                noise = estimation
                x_intermediate = (x_current - noise * self.diffusion.extract(self.diffusion.sqrt_one_minus_alphas_cumprod, t, x.shape)) / self.diffusion.extract(self.diffusion.sqrt_alphas_cumprod, t, x.shape)
            else:
                x_intermediate = estimation
                noise = (x_current -  self.diffusion.extract(self.diffusion.sqrt_alphas_cumprod, t, x.shape) * x_intermediate) / self.diffusion.extract(self.diffusion.sqrt_one_minus_alphas_cumprod, t, x.shape)

            sqrt_alpha_prev_time = self.diffusion.extract(self.diffusion.sqrt_alphas_cumprod, t-1, x.shape)
            sqrt_alpha_curr_time = self.diffusion.extract(self.diffusion.sqrt_alphas_cumprod, t, x.shape)
            
            sigma = ((1 - sqrt_alpha_prev_time.pow(2)) / (1 - sqrt_alpha_curr_time.pow(2))).sqrt() * \
                        (1 - sqrt_alpha_curr_time.pow(2) / sqrt_alpha_prev_time.pow(2)).sqrt()
            if sigma.isnan().any():
                sigma = torch.zeros_like(sigma, device= noise.device)
            sigma = sigma * eta

            if t == 1:
                sigma = torch.zeros_like(sigma, device= noise.device)
            # print(f't: {t.item()}, sigma: {sigma.item()}')

            scaled_estimate =  sqrt_alpha_prev_time * x_intermediate
            deterministic_noise = torch.sqrt(1 - sqrt_alpha_prev_time.pow(2) - sigma.pow(2)) * noise
            stochastic_noise = sigma * torch.randn_like(x_intermediate)

            # DPS guidance
            difference = self.masked_kspace - self.fwd(x_intermediate)
            norm = difference.square().mean()
            norm_grad = torch.autograd.grad(norm, x_current)[0]

            x_current = scaled_estimate + deterministic_noise + stochastic_noise
            x_current = x_current - scale/norm * norm_grad

            t -= 1

        return x_current

    def reconstruct(self):
        """
        Perform the reconstruction process.

        Returns:
            torch.Tensor: The reconstructed image.
        """
        b, _, f, coil, _, _ = self.masked_kspace.shape
        _, _, h, w = self.sensitivities.shape

        t = torch.tensor([int((self.diffusion.timesteps-1))], device=self.masked_kspace.device)
        x = torch.randn((b, 2, f, h, w), device=self.masked_kspace.device)

        recon = self.DPS_sample(x, t, eta=self.args.model['eta'], scale=self.args.model['scale'])

        recon = recon.detach()
        recon = torch.view_as_complex(recon.squeeze(0).transpose(0, 3).contiguous()).permute(1, 2, 0)

        return recon