import argparse
from pathlib import Path
from types import SimpleNamespace

import torch
import numpy as np

from torch.utils import data

from cardiac_diffusion.dataset import CMRxProcessed, OCMRxProspective
from cardiac_diffusion.utils import tensor_to_gif
from cardiac_diffusion.metrics import compute_metrics
from cardiac_diffusion.mri import kspace2image

from cardiac_diffusion.pddr import ReconstructionPDDR
from baselines.lps import LowRankPlusSparse
from baselines.fmlp import ReconstructionFMLP
from baselines.tdip import ReconstructionTDIP
from baselines.dps import ReconstructionDPS
from baselines.dstdm import ReconstructionDSTDM


def zero_filled_reconstruction(args, masked_kspace, mask, sensitivities):
    ''' Zero-filled reconstruction / Pseudo-inverse '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    masked_kspace, sensitivities, mask = masked_kspace.to(device=device), sensitivities.to(device=device), mask.to(device=device)
    # Push to cuda only for fair metrics wrt memory and speed

    zf_kspace_cx = masked_kspace.unsqueeze(0)
    sensitivities = sensitivities.unsqueeze(0)

    zf_image = kspace2image(zf_kspace_cx, sens_maps=sensitivities.unsqueeze(1), fdim=(3, 4), cdim=2).squeeze(0)

    return zf_image


def pddr_reconstruct(args, masked_kspace, mask, sensitivities):
    ''' Piecewise Dynamic Diffusion Regularization (PDDR) reconstruction '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    masked_kspace, sensitivities, mask = masked_kspace.to(device=device), sensitivities.to(device=device), mask.to(device=device)

    ## Setup PDDR model and run experiment
    pddr = ReconstructionPDDR(args, masked_kspace, mask, sensitivities)
    reconstruction = pddr.reconstruct()

    return reconstruction


def lps_reconstruct(args, masked_kspace, mask, sensitivities):
    ''' L+S reconstruction '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    masked_kspace, sensitivities, mask = masked_kspace.to(device=device), sensitivities.to(device=device), mask.to(device=device)

    ## Hyperparameters
    intermediate_path = Path(args.output_path) / 'intermediate_results'
    max_iter = 600
    intermediate_save_frequency = 0
    if intermediate_save_frequency > 0:
        intermediate_path.mkdir(exist_ok=True)
    lambda_l = args.model['lambda_l'] # 0.1
    lambda_s = args.model['lambda_s'] # 0.01

    ## Setup L+S model and run experiment
    lps = LowRankPlusSparse(masked_kspace, sensitivities, mask, intermediate_path, intermediate_save_frequency).to_device(device)
    reconstruction = lps.run(max_iter=max_iter, lambda_l=lambda_l, lambda_s=lambda_s, tol=1e-6)

    return reconstruction


def dps_reconstruct(args, masked_kspace, mask, sensitivities):
    ''' Diffusion Posterior Sampling (DPS) reconstruction '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    masked_kspace, sensitivities, mask = masked_kspace.to(device=device), sensitivities.to(device=device), mask.to(device=device)

    ## Hyperparameters
    if args.inference_config:
        inference_config = ReconstructionDPS.load_yaml(args.inference_config)
        for key, value in inference_config.items():
            setattr(args, key, value)

    ## Setup DPS model and run experiment
    dps = ReconstructionDPS(args, masked_kspace, mask, sensitivities)
    reconstruction = dps.reconstruct()

    return reconstruction


def dstdm_reconstruct(args, masked_kspace, mask, sensitivities):
    ''' dSTDM reconstruction '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    masked_kspace, sensitivities, mask = masked_kspace.to(device=device), sensitivities.to(device=device), mask.to(device=device)

    if getattr(args, "model", None) is None:
        args.model = {
            'path': '/mnt/hdd_pool_zion/userdata/florian/cine_models/dSTDM',
            'checkpoint': 'model.pt',
            'lambda': 0.5,
            'rho': 1.0,
        }

    dstdm = ReconstructionDSTDM(args, masked_kspace, mask, sensitivities)
    reconstruction = dstdm.reconstruct()

    return reconstruction


def fmlp_reconstruct(args, masked_kspace, mask, sensitivities):
    ''' FMLP reconstruction '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    masked_kspace, sensitivities, mask = masked_kspace.to(device=device), sensitivities.to(device=device), mask.to(device=device)

    param = SimpleNamespace()
    param.experiment = SimpleNamespace()
    param.data  = SimpleNamespace()
    param.hp = SimpleNamespace()
    param.fmlp = SimpleNamespace()
    param.optimizer = SimpleNamespace()
    param.tvloss = SimpleNamespace()

    ## Basic hyperparameters
    param.hp.num_iter = args.model['num_iter'] # 1000
    param.hp.lambda_tv = 0.

    ## Experiment configuration
    output_path = Path(args.output_path) / 'trained_models'
    output_path.mkdir(exist_ok=True)
    param.experiment.output_path = output_path
    param.experiment.model_save_frequency = 1000
    param.experiment.video_evaluation_frequency = 1000
    param.experiment.ser_validation = args.model['ser_early_stopping'] # True
    param.experiment.validation_evaluation_frequency = 50
    param.experiment.epochs_after_last_highscore = 200
    param.experiment.minimal_epochs = 200

    ## Dataset Configuration
    Nk, Nc, Ny, Nx = masked_kspace.shape
    param.data.Nk = Nk
    param.data.Nc = Nc
    param.data.Nx = Nx
    param.data.Ny = Ny

    if args.data['ocmr_realtime']:

        print(args.data['FOV'])
        print(args.data['TRes'], args.data['TRes'][0], float(args.data['TRes'][0].strip('[]')))

        param.data.tres = float(args.data['TRes'][0].strip('[]')) / 1000  # convert from ms to s
        param.data.frame_times = torch.tensor([param.data.tres*(i+0.5) for i in range(Nk)])
        param.data.frame_rate = 1 / param.data.tres

        param.data.fov = { # FOV [m]
            "y": float(args.data['FOV'][1] / 1000),  # convert from mm to m
            "x": float(args.data['FOV'][0] / 1000)
        } 

        print(param.data.fov, param.data.frame_times, param.data.frame_rate)

    else:
        # raise NotImplementedError("FMLP fov and tr parameters not implemented yet.")

        param.data.frame_times = torch.tensor([0.025 + 0.05*i for i in range(Nk)])
        param.data.frame_rate = 1.0 / 0.05
        param.data.fov = { # FOV [m]
            "y": 0.0015*Ny,
            "x": 0.0015*Nx
        } 

    ## FMLP parameters
    sx = args.model['sx'] # 20.  # 30. # 60. # 100.
    param.fmlp.spatial_in_features = 2
    param.fmlp.spatial_fmap_width = 512
    param.fmlp.spatial_coordinate_scales = [sx, sx] # spatial coordinate scale in [1/m]
                
    st = args.model['st'] # 1.            
    param.fmlp.temporal_in_features = 1
    param.fmlp.temporal_fmap_width = 128
    param.fmlp.temporal_coordinate_scales = [st] # temporal coordinate scale in [1/s]

    param.fmlp.mlp_width = 512
    param.fmlp.mlp_sigma = 0.01
    param.fmlp.mlp_scale = 1.
    param.fmlp.mlp_hidden_layers = 7
    param.fmlp.mlp_hidden_bias = True

    param.fmlp.mlp_out_features = 2
    param.fmlp.mlp_final_sigma = 0.01
    param.fmlp.mlp_final_bias = True

    param.fmlp.out_scale = args.model['out_scale'] # 120.
    
    ## optimizer parameters
    param.optimizer.weight_decay = 0
    param.optimizer.lr = 2e-4

    ## tv regularization parameters
    param.tvloss.num_elements = param.data.Nk
    param.tvloss.mode = "real_imag"
    param.tvloss.directionality = "both"
    param.tvloss.normalize = "false"

    ## Setup FMLP model and run experiment
    model = ReconstructionFMLP(param)
    model.train(masked_kspace, sensitivities, mask)
    reconstruction = model.reconstruct()

    return reconstruction


def tdip_reconstruct(args, masked_kspace, mask, sensitivities):
    ''' T-DIP reconstruction '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    masked_kspace, sensitivities, mask = masked_kspace.to(device=device), sensitivities.to(device=device), mask.to(device=device)

    param = SimpleNamespace()
    param.experiment = SimpleNamespace()
    param.data  = SimpleNamespace()
    param.hp = SimpleNamespace()
    param.decoder = SimpleNamespace()
    param.optimizer = SimpleNamespace()
    param.trajectory = SimpleNamespace()

    ## Basic hyperparameters
    param.hp.num_iter = args.model['num_iter'] # 1000
    param.hp.lambda_tv = 0.

    ## Experiment configuration
    output_path = Path(args.output_path) / 'trained_models'
    output_path.mkdir(exist_ok=True)
    param.experiment.output_path = output_path
    param.experiment.model_save_frequency = 1000 # 50
    param.experiment.video_evaluation_frequency = 1000 # 50
    param.experiment.ser_validation = args.model['ser_early_stopping'] # True
    param.experiment.validation_evaluation_frequency = 50
    param.experiment.epochs_after_last_highscore = 200
    param.experiment.minimal_epochs = 500

    ## Dataset Configuration
    Nk, Nc, Ny, Nx = masked_kspace.shape
    param.data.Nk = Nk
    param.data.Nc = Nc
    param.data.Nx = Nx
    param.data.Ny = Ny

    if args.data['ocmr_realtime']:

        param.data.tres = float(args.data['TRes'][0].strip('[]')) / 1000  # convert from ms to s

        param.data.frame_times = torch.tensor([param.data.tres*(i+0.5) for i in range(Nk)])
        param.data.frame_rate = 1 / param.data.tres
        param.trajectory.p = Nk / 20.0 # estimate, as no ecg estimate avialable

    else:
        # raise NotImplementedError("T-DIP fov and tr parameters not implemented yet.")

        param.data.frame_times = torch.tensor([0.025 + 0.05*i for i in range(Nk)])
        param.data.frame_rate = 1.0 / 0.05
        param.trajectory.p = Nk / 12.0 # estimate, as no ecg estimate avialable (CMRxRecon consists of 12 frames)

    ## Decoder parameters
    param.decoder.in_features = 3
    param.decoder.out_features = 2
    param.decoder.out_size = [param.data.Ny, param.data.Nx]
    param.decoder.map_net_out_size = [6, 9]
    param.decoder.num_stages = 6 # 5
    param.decoder.num_conv = 2
    param.decoder.conv_channels = 256
    param.decoder.conv_bias = False
    param.decoder.output_scaling = 32.

    param.trajectory.type = "helix"
    param.trajectory.L = 3
    param.trajectory.z_slack = [0.5]
    # param.trajectory.p = param.data.cardiac_cycles[param.data.Nk-1] + param.data.cardiac_phases[param.data.Nk-1] - param.data.cardiac_phases[0]
    param.trajectory.equal_frame_size = True # False
    
    ## optimizer parameters
    param.optimizer.weight_decay = 0
    param.optimizer.lr = 1e-4

    ## Setup TDIP model and run experiment
    model = ReconstructionTDIP(param)
    model.train(masked_kspace, sensitivities, mask)
    reconstruction = model.reconstruct()

    return reconstruction


def get_data(args):
    slice_idx = args.data['slice_idx']
    slice_idx = 0 if slice_idx is None else slice_idx

    if args.data['ocmr_realtime']:
        ds = OCMRxProspective(data_path=args.data['path'])

        masked_kspace, mask, sensitivities, fname, finfo = ds[slice_idx]

        ground = None
        args.data['FOV'] = finfo['FOV']
        args.data['TRes'] = finfo['TRes']

        masked_kspace = torch.view_as_complex(masked_kspace.transpose(0, 4).contiguous())
        masked_kspace = masked_kspace.permute(1, 2, 3, 0)

        print(f'Loaded data {fname}.')
        return masked_kspace, mask, sensitivities, ground, fname
    
    else:
        ds = CMRxProcessed(
            data_path=args.data['path'], 
            repeat_cycle=args.data['repeat_cycle'],
            undersample=args.data['undersample'],
            undersampling_factor=args.data['undersampling_factor'],
            undersampling_type=args.data['undersampling_type'],
            alpha=args.data['alpha'],
            seed=args.data['seed'],
            return_fname=True
            )
        
        masked_kspace, mask, sensitivities, ground, fname = ds[slice_idx]

        masked_kspace = torch.view_as_complex(masked_kspace.transpose(0, 4).contiguous())
        masked_kspace = masked_kspace.permute(1, 2, 3, 0)
        ground = ground.unsqueeze(1)

        print(f'Loaded data {fname}.')
        return masked_kspace, mask, sensitivities, ground, fname


def evaluate_reconstruction(args, recon, ground, store_results=True):

    if store_results:    
        torch.save(recon, f'{args.output_path}/reconstruction.pt')

    if ground is None:
        metric_recon = abs(recon).unsqueeze(1)
    else:
        mag_ground = abs(ground)
        mag_recon = abs(recon).unsqueeze(1)

        ssims, psnrs, nmses, metric_recon = compute_metrics(mag_ground, mag_recon)
        print(f'Baseline Reconstruction -- SSIM: {np.mean(ssims):.4f}, PSNR: {np.mean(psnrs):.2f}, NMSE: {np.mean(nmses):.6f}')
        metric_recon = metric_recon

    if store_results:
        tensor_to_gif(metric_recon, f'{args.output_path}/reconstruction.gif', duration=50) # 112

    if ground is None: 
        return None, None, None
    else:
        return ssims, psnrs, nmses


def choose_baseline(baseline):
    # choose the baseline
    if baseline == 'PDDR':
        base = pddr_reconstruct
    elif baseline == 'SDR':
        base = pddr_reconstruct
    elif baseline == 'L+S':
        base = lps_reconstruct
    elif baseline == 'DPS':
        base = dps_reconstruct
    elif baseline == 'dSTDM':
        base = dstdm_reconstruct
    elif baseline == 'FMLP':
        base = fmlp_reconstruct
    elif baseline == 'TDIP':
        base = tdip_reconstruct
    elif baseline == 'ZF':
        base = zero_filled_reconstruction
    else:
        raise ValueError(f'Unknown baseline: {baseline}')
    
    return base


def main():
    parser = argparse.ArgumentParser()
    #
    parser.add_argument("-b", "--base", type=Path, required=True, metavar="/path/to/base_directory", help='Path to the base_directory.')
    parser.add_argument('-i', '--inference_config', type=str, default='configs/inference.yaml', metavar="/path/to/inference.yaml", 
                        help='Path to the inference configuration file. [Optional] Defaults to: configs/inference.yaml.')
    parser.add_argument('--baseline', default='PDDR', type=str, metavar="method", help='Change the reconstruction method.')
    parser.add_argument('--write_tensorboard', action='store_true', help='Write intermediate results to TensorBoard')
    #
    args, unknown = parser.parse_known_args()

    # choose baseline
    base = choose_baseline(args.baseline)

    # Add additional parameters to args
    inference_config = ReconstructionPDDR.load_yaml(args.inference_config)
    for key, value in inference_config.items():
        setattr(args, key, value)

    # Merge data directories with base path
    args.output_path = args.base / args.output_path
    args.data['path'] = args.base / args.data['path']
    args.model['path'] = args.base / args.model['path']

    # get data
    masked_kspace, mask, sensitivities, ground, fname = get_data(args)

    # compute baseline reconstruction
    recon = base(args, masked_kspace, mask, sensitivities)
    evaluate_reconstruction(args, recon, ground)


if __name__ == "__main__":

    print(f'CUDA: {torch.cuda.is_available(), torch.cuda.device_count()}')
    main()
