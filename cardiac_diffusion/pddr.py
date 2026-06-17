import torch
import yaml
from pathlib import Path
from copy import deepcopy

from cardiac_diffusion.diffusion import Diffusion
from cardiac_diffusion.variational import Variation
from cardiac_diffusion.unet import SpatioTemporalUNetModel, UNetModel

from cardiac_diffusion.mri import kspace2image

class ReconstructionPDDR:
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
        diffusion_config_path = Path(args.model['path']) / 'configs' / 'diffusion.yaml'
        diffusion_config = self.load_yaml(diffusion_config_path)
        for key, value in diffusion_config.items():
            setattr(args, key, value)

        # Set output directories
        args.model_path = Path(args.model['path']) / 'models' / args.model['checkpoint']
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

        # Build the variation model
        self.variation = Variation(
            prior_model=self.diffusion,
            measurements=self.masked_kspace,
            mask=self.mask,
            sensitivities=self.sensitivities,
            unet_type=args.model['unet_type'],
            sparse_representation=False,
            output_dir=args.output_dir,
            model_path=args.model_path,
            **args.variation,
            **args.optimizer,
            write_tensorboard=(args.write_tensorboard)
        )

    @staticmethod
    def load_yaml(file_path):
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file)
        return config

    @staticmethod
    def save_yaml(config, file_path):
        with open(file_path, 'w') as file:
            yaml.safe_dump(config, file)

    def initialize_image(self, initialization='zero_filled', sparse_representation=False):
        b, _, f, coil, _, _ = self.masked_kspace.shape
        _, _, h, w = self.sensitivities.shape

        if initialization == 'zero_filled':
            masked_kspace_cx = torch.complex(self.masked_kspace[:, 0], self.masked_kspace[:, 1])

            if sparse_representation:
                zf_kspace_cx = torch.zeros((b, f, coil, h, w), dtype=masked_kspace_cx.dtype, device=self.masked_kspace.device)
                ky_indices = self.mask[:, :, :, :, 1]
                ky_expanded = ky_indices.unsqueeze(2).expand(-1, -1, masked_kspace_cx.shape[2], -1, -1) 
                zf_kspace_cx = zf_kspace_cx.scatter(3, ky_expanded, masked_kspace_cx)
            else:
                zf_kspace_cx = masked_kspace_cx

            zf_image = kspace2image(zf_kspace_cx, sens_maps=self.sensitivities.unsqueeze(1), fdim=(3, 4), cdim=2)
            init = torch.stack((zf_image.real, zf_image.imag), dim=1).cuda()
        elif initialization == 'zeros':
            init = torch.zeros((b, 2, f, h, w), device=self.masked_kspace.device)
        elif initialization == 'ones':
            init = torch.ones((b, 2, f, h, w), device=self.masked_kspace.device)
        else:
            init = torch.randn((b, 2, f, h, w), device=self.masked_kspace.device)

        return init


    def reconstruct(self):
        """
        Perform the reconstruction process.

        Returns:
            torch.Tensor: The reconstructed image.
        """
        # Initialize the image
        init = self.initialize_image(
            initialization=self.args.model['initialization']
        )
        x = torch.nn.Parameter(
            deepcopy(init), requires_grad=True
        )

        # Fit the variation model
        self.variation.fit(x)

        recon = x.detach()
        recon = torch.view_as_complex(recon.squeeze(0).transpose(0, 3).contiguous()).permute(1, 2, 0)

        return recon