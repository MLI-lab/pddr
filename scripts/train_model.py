import os
import argparse
import yaml
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from cardiac_diffusion.unet import SpatioTemporalUNetModel, UNetModel
from cardiac_diffusion.diffusion import Diffusion
from cardiac_diffusion.trainer import Trainer

import socket
from contextlib import closing


def find_free_port():
    """ https://stackoverflow.com/questions/1365265/on-localhost-how-do-i-pick-a-free-port-number &
        https://stackoverflow.com/questions/66498045/how-to-solve-dist-init-process-group-from-hanging-or-deadlocks
    """

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('localhost', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])

def ddp_setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def ddp_cleanup():
    dist.destroy_process_group()

def train(rank, world_size, args):
    ddp_setup(rank, world_size)

    if args.data['unet_type'] == 'UNetModel':
        restoration_fn = UNetModel(
            **args.UNetModel
        ).cuda(rank)
    elif args.data['unet_type'] == 'SpatioTemporalUNetModel':
        restoration_fn = SpatioTemporalUNetModel(
            **args.SpatioTemporalUNetModel
        ).cuda(rank)
    else:
        raise ValueError(f"Unknown unet_type: {args.data['unet_type']}")

    diffusion = Diffusion(
        restoration_fn=restoration_fn,
        **args.diffusion
    ).cuda(rank)

    diffusion = DDP(diffusion, device_ids=[rank])

    trainer = Trainer(
        diffusion_model=diffusion,
        **args.output_dirs,
        **args.data,
        **args.params,
        **args.sample,
        **args.ema
    )

    trainer.train()

    ddp_cleanup()

def load_yaml(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def save_yaml(config, file_path):
    with open(file_path, 'w') as file:
        yaml.safe_dump(config, file)

def set_output_dirs(args):

    output_dir = args.base / 'models' / args.output['model_name']

    try:
        output_dir.mkdir(exist_ok=False)
    except FileExistsError:
        print(f"The directory '{output_dir}' already exists!")
        if args.data['resume_training']:
            print("Resuming training from previous checkpoint.")
        else:
            response = input("Are you sure you want to override previously trained models? (y/n): ").strip().lower()
            if response == 'y':
                print("Potentially overriding previously trained models.")
            else:
                print("Please redefine the output name to avoid overriding.")
                exit(1)

    config_dir = output_dir / 'configs'
    config_dir.mkdir(exist_ok=True)
    
    log_dir = output_dir / 'logs'
    log_dir.mkdir(exist_ok=True)
    
    model_dir = output_dir / 'models'
    model_dir.mkdir(exist_ok=True)

    # Add output_dirs parameter to args
    args.output_dirs = {
        'log_dir': log_dir,
        'model_dir': model_dir
    }

    return config_dir

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--base", type=Path, required=True, metavar="/path/to/base_directory", help='Path to the base_directory.')
    parser.add_argument('-d', '--diffusion_config', type=str, default='configs/diffusion.yaml', metavar="/path/to/diffusion.yaml", 
                        help='Path to the diffusion model configuration file. [Optional] Defaults to: configs/diffusion.yaml.')
    parser.add_argument('-t', '--training_config', type=str, default='configs/training.yaml', metavar="/path/to/training.yaml", 
                        help='Path to the training configuration file. [Optional] Defaults to: configs/training.yaml.')
    args = parser.parse_args()

    # Load configs from yaml files
    diffusion_config = load_yaml(args.diffusion_config)
    training_config = load_yaml(args.training_config)

    # Add additional parameters to args
    for key, value in diffusion_config.items():
        setattr(args, key, value)
    for key, value in training_config.items():
        setattr(args, key, value)

    # Set output directories
    config_dir = set_output_dirs(args)

    # Save configs for reproducibility
    save_yaml(diffusion_config, config_dir / 'diffusion.yaml')
    if args.data['model_path'] is not None:
        name = Path(args.data['model_path']).stem
        save_yaml(training_config, config_dir / f'training-{name}.yaml')
    else:
        save_yaml(training_config, config_dir / 'training.yaml')

    # Merge data directories with base path
    if isinstance(args.data['train_path'], (list, tuple)):
        args.data['train_path'] = [args.base / p for p in args.data['train_path']]
    else:
        args.data['train_path'] = args.base / args.data['train_path']

    if args.data['val_path'] is not None:
        args.data['val_path'] = args.base / args.data['val_path']

    # run training
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = find_free_port() # '12356'
    world_size = torch.cuda.device_count()
    
    mp.spawn(train, args=(world_size, args), nprocs=world_size, join=True)


if __name__ == "__main__":

    print(f'CUDA: {torch.cuda.is_available(), torch.cuda.device_count()}')
    main()
