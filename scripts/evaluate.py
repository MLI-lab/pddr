import argparse
import yaml
import json
import csv
from itertools import product
from pathlib import Path
import numpy as np
import time

import torch
from torch.utils import data
from copy import deepcopy

from cardiac_diffusion.dataset import CMRxTestset, OCMRxProspective
from cardiac_diffusion.metrics import split_mask_and_kspace, compute_ser, compute_ttv
from cardiac_diffusion.utils import tensor_to_gif
from reconstruct import choose_baseline, evaluate_reconstruction

def reconstruct(args):

    base = choose_baseline(args.model['baseline'])
    dl = get_dataloader(args)

    ssim, psnr, nmse, times, mems, sers, ttvs = [], [], [], [], [], [], []
    for sample_idx in range(len(dl)):

        masked_kspace, mask, sensitivities, ground, fname = get_data(args, dl)

        if args.experiment['ser']:
            ser_split_seed = int(args.data['seed']) + int(sample_idx)
            mask, masked_kspace, validation_mask, validation_kspace = split_mask_and_kspace(
                mask.unsqueeze(0),
                masked_kspace.unsqueeze(0),
                validation_lines=1,
                sparse_representation=False,
                seed=ser_split_seed,
            )
            mask, masked_kspace, validation_mask, validation_kspace = mask.squeeze(0), masked_kspace.squeeze(0), validation_mask.squeeze(0), validation_kspace.squeeze(0)
            print('Using validation lines for SER computation.')

        torch.cuda.reset_peak_memory_stats()
        start = time.time()
        recon = base(args, masked_kspace, mask, sensitivities)
        end = time.time()
        timer = end - start
        mem = torch.cuda.max_memory_reserved() / 1024 ** 2

        if args.experiment['store_all']:
            out = args.output_path / f'{fname[0]}' / 'output'
            out.mkdir(exist_ok=True, parents=True)
            torch.save(recon, out / f'{args.model["baseline"]}.pt')
            rec = abs(recon).unsqueeze(1)
            tensor_to_gif(rec, out.parent / f'{args.model["baseline"]}.gif', duration=50)
            if ground is not None:
                torch.save(ground, out / 'ground.pt')
            if args.experiment['ser']:
                torch.save(mask, out.parent / 'training_mask.pt')
                torch.save(masked_kspace, out.parent /' training_kspace.pt')
                torch.save(validation_mask, out.parent / 'validation_mask.pt')
                torch.save(validation_kspace, out.parent / 'validation_kspace.pt')
                torch.save(sensitivities, out.parent / 'sensitivities.pt')

        ssims, psnrs, nmses = evaluate_reconstruction(args, recon, ground)
        if args.experiment['ser']:
            ser = compute_ser(recon.cpu(), validation_mask, validation_kspace, sensitivities)
            sers.append(ser)
            ttv = compute_ttv(recon.cpu())
            ttvs.append(ttv)
        else:
            ser = None
            sers = None
            ttv = None
            ttvs = None

        m_ssim, m_psnr, m_nmse, _, _, _, _ = save_metrics_csv(args.output_path / 'metrics.csv', fname, ssims, psnrs, nmses, timer, mem, ser, ttv)

        if m_ssim: ssim.append(m_ssim) 
        else: ssim = None
        if m_psnr: psnr.append(m_psnr) 
        else: psnr = None
        if m_nmse: nmse.append(m_nmse) 
        else: nmse = None
        times.append(timer)
        mems.append(mem)
            
    ssim, psnr, nmse, timer, max_mem, ser, ttv = save_metrics_csv(args.output_path / 'metrics.csv', ['OVERALL'], ssim, psnr, nmse, times, mems, sers, ttvs)

    return ssim, psnr, nmse, timer, max_mem, ser, ttv


def get_dataloader(args):

    if args.data['ocmr_realtime']:
        ds = OCMRxProspective(data_path=args.testsets[args.experiment['testset']])
    else:
        ds = CMRxTestset(
            data_path=args.testsets[args.experiment['testset']], 
            repeat_cycle=args.data['repeat_cycle'],
            undersample=args.data['undersample'],
            undersampling_factor=args.data['undersampling_factor'],
            undersampling_type=args.data['undersampling_type'],
            alpha=args.data['alpha'],
            seed=args.data['seed']
            )
    
    print(f'Dataset length: {len(ds)}')
        
    dl = iter(
        data.DataLoader(ds,
                        batch_size=1,
                        shuffle=False,
                        pin_memory=False,
                        num_workers=4,
                        drop_last=False,
                        persistent_workers=False
                        ) 
    )

    return dl

def get_data(args, dl):
        if args.data['ocmr_realtime']:
            masked_kspace, mask, sensitivities, fname, finfo = next(dl)
            ground = None

            args.data['FOV'] = finfo['FOV']
            args.data['TRes'] = finfo['TRes']
        else:
            masked_kspace, mask, sensitivities, ground, fname = next(dl)
            ground = ground.squeeze(0).unsqueeze(1)

        masked_kspace = torch.view_as_complex(masked_kspace.squeeze(0).transpose(0, 4).contiguous())
        masked_kspace = masked_kspace.permute(1, 2, 3, 0)
        mask = mask.squeeze(0)
        sensitivities = sensitivities.squeeze(0)

        return masked_kspace, mask, sensitivities, ground, fname

def save_metrics_csv(metrics_path, fname, ssims, psnrs, nmses, timer, mem, ser, ttv):

    if ssims is None:
        m_ssim = None
        std_ssim = None
        ssim = 'SSIM: -'
    else:
        m_ssim = np.mean(ssims)
        std_ssim = np.std(ssims)
        ssim = f'SSIM: {m_ssim:.4f} +- {std_ssim:.4f}'

    if psnrs is None:
        m_psnr = None
        std_psnr = None
        psnr = 'PSNR: -'
    else:
        m_psnr = np.mean(psnrs)
        std_psnr = np.std(psnrs)
        psnr = f'PSNR: {m_psnr:.2f} +- {std_psnr:.2f} dB'

    if nmses is None:
        m_nmse = None
        std_nmse = None
        nmse = 'NMSE: -'
    else:
        m_nmse = np.mean(nmses)
        std_nmse = np.std(nmses)
        nmse = f'NMSE: {m_nmse:.4f} +- {std_nmse:.4f}'

    if ser is None:
        m_ser = None
        std_ser = None
        ser = 'SER: -'
    else:
        m_ser = np.mean(ser)
        std_ser = np.std(ser)
        ser = f'SER: {m_ser:.2f} +- {std_ser:.2f} dB'

    if ttv is None:
        m_ttv = None
        std_ttv = None
        ttv = 'TTV: -'
    else:
        m_ttv = np.mean(ttv)
        std_ttv = np.std(ttv)
        ttv = f'TTV: {m_ttv:.4f} +- {std_ttv:.4f}'

    if isinstance(timer, list):
        m_timer = np.mean(timer)
        std_timer = np.std(timer)
        timer = f'TIME: {m_timer:.3f} +- {std_timer:.3f} s'
    else:
        m_timer = timer
        std_timer = 0
        timer = f'TIME: {m_timer:.3f} s'

    if isinstance(mem, list):
        m_mem = np.mean(mem)
        std_mem = np.std(mem)
        mem = f'MEMORY: {m_mem:.2f} +- {std_mem:.2f} MB'
    else:
        m_mem = mem
        std_mem = 0
        mem = f'MEMORY: {m_mem:.2f} MB'


    with open(metrics_path, 'a') as f:
        f.write(f'{fname[0]}\n')
        f.write(f';SSIM;{m_ssim};{std_ssim};{ssim}\n') 
        f.write(f';PSNR;{m_psnr};{std_psnr};{psnr}\n')
        f.write(f';NMSE;{m_nmse};{std_nmse};{nmse}\n')
        f.write(f';SER;{m_ser};{std_ser};{ser}\n') 
        f.write(f';TTV;{m_ttv};{std_ttv};{ttv}\n') 
        f.write(f';time;{m_timer};{std_timer};{timer}\n') 
        f.write(f';memory;{m_mem};{std_mem};{mem}\n\n') 

    print(f'{fname[0]}: {ssim}, {psnr}, {nmse}, {timer}, {mem}, {ser}, {ttv}')
    
    return m_ssim, m_psnr, m_nmse, m_timer, m_mem, m_ser, m_ttv

def load_yaml(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def save_yaml(config, file_path):
    with open(file_path, 'w') as file:
        yaml.safe_dump(config, file)

def recursive_iterations(i, ablations, operation):
    for j, (k, v) in enumerate(ablations.items()):
        if j == i+1:
            for param in v:
                operation(j, k, param)
                recursive_iterations(j, ablations, operation)


def parse_overall_from_csv(csv_path):
    """Parse the OVERALL row from a metrics.csv, returning a dict of metric -> mean value."""
    metrics = {}
    with open(csv_path, 'r') as f:
        lines = f.readlines()
    
    in_overall = False
    for line in lines:
        line = line.strip()
        if line.startswith('OVERALL'):
            in_overall = True
            continue
        if in_overall and line.startswith(';'):
            parts = line.split(';')
            # format: ;METRIC;mean;std;printable
            if len(parts) >= 3:
                metric_name = parts[1].strip()
                mean_val = parts[2].strip()
                if mean_val and mean_val != 'None':
                    try:
                        metrics[metric_name] = float(mean_val)
                    except ValueError:
                        pass
        elif in_overall and line and not line.startswith(';'):
            # next file entry, stop
            break
    return metrics


def write_seed_summary(args):
    """Write summary.csv files aggregating results across seeds for each condition group.
    
    Works for arbitrary ablation grids. Identifies 'data.seed' as the replication axis
    and groups by all other ablation dimensions.
    """
    ablations = args.experiment['ablations']
    ablation_keys = list(ablations.keys())
    
    # Find the seed key
    seed_key = 'data.seed'
    if seed_key not in ablations:
        print('No data.seed in ablations — skipping cross-seed summary.')
        return
    
    seed_values = ablations[seed_key]
    seed_key_index = ablation_keys.index(seed_key)
    
    # Non-seed ablation keys and their values
    condition_keys = [k for k in ablation_keys if k != seed_key]
    condition_values = [ablations[k] for k in condition_keys]
    
    base_output = Path(args.experiment['output']) / args.experiment['name']
    
    # Iterate over every combination of non-seed ablation parameters
    if condition_keys:
        condition_combos = list(product(*condition_values))
    else:
        condition_combos = [()]  # single group if seed is the only ablation
    
    for combo in condition_combos:
        # Build a dict mapping ablation key -> value for this condition group
        condition_map = dict(zip(condition_keys, combo))
        
        seed_metrics = []  # list of dicts: {seed, SSIM, PSNR, NMSE, ...}
        
        for seed_val in seed_values:
            # Build the output directory path in ablation key order
            dir_parts = []
            for k in ablation_keys:
                if k == seed_key:
                    dir_parts.append(str(seed_val))
                else:
                    dir_parts.append(str(condition_map[k]))
            
            csv_path = base_output / '/'.join(dir_parts) / 'metrics.csv'
            
            if not csv_path.exists():
                print(f'Warning: {csv_path} not found, skipping seed {seed_val}.')
                continue
            
            overall = parse_overall_from_csv(csv_path)
            if overall:
                overall['seed'] = seed_val
                seed_metrics.append(overall)
        
        if not seed_metrics:
            continue
        
        # Determine summary output path
        # Place summaries under seed_summaries/ to avoid confusion with run directories
        # when data.seed is not the last ablation key
        summary_dir_parts = []
        for k in ablation_keys:
            if k != seed_key:
                summary_dir_parts.append(str(condition_map[k]))
        
        if summary_dir_parts:
            summary_path = base_output / 'seed_summaries' / '/'.join(summary_dir_parts) / 'summary.csv'
        else:
            summary_path = base_output / 'seed_summaries' / 'summary.csv'
        
        summary_path.parent.mkdir(exist_ok=True, parents=True)
        
        # Collect all metric names (excluding 'seed')
        all_metric_names = []
        for m in seed_metrics:
            for mk in m.keys():
                if mk != 'seed' and mk not in all_metric_names:
                    all_metric_names.append(mk)
        
        # Write summary CSV
        with open(summary_path, 'w', newline='') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow(['seed'] + all_metric_names)
            
            for sm in seed_metrics:
                row = [sm.get('seed', '')]
                for mn in all_metric_names:
                    val = sm.get(mn)
                    row.append(f'{val:.4f}' if val is not None else '-')
                writer.writerow(row)
            
            # Mean ± std row
            mean_row = ['mean']
            std_row = ['std']
            summary_row = ['mean+/-std']
            for mn in all_metric_names:
                vals = [sm[mn] for sm in seed_metrics if mn in sm and sm[mn] is not None]
                if vals:
                    m = np.mean(vals)
                    s = np.std(vals)
                    mean_row.append(f'{m:.4f}')
                    std_row.append(f'{s:.4f}')
                    summary_row.append(f'{m:.4f} +/- {s:.4f}')
                else:
                    mean_row.append('-')
                    std_row.append('-')
                    summary_row.append('-')
            
            writer.writerow([])
            writer.writerow(mean_row)
            writer.writerow(std_row)
            writer.writerow(summary_row)
        
        condition_str = ', '.join(f'{k}={condition_map[k]}' for k in condition_keys) if condition_keys else 'all'
        print(f'Seed summary written to {summary_path} ({condition_str}, {len(seed_metrics)} seeds)')


def main():
    parser = argparse.ArgumentParser()
    #
    parser.add_argument("-b", "--base", type=Path, required=True, metavar="/path/to/base_directory", help='Path to the base_directory.')
    parser.add_argument('-e', '--experiment_config', type=str, default='configs/experiment.yaml', metavar="/path/to/experiment.yaml", 
                        help='Path to the experiment configuration file. [Optional] Defaults to: configs/experiment.yaml.')
    #
    args = parser.parse_args()

    # Load configs from yaml files and add additional parameters to args
    experiment_config = load_yaml(args.experiment_config)
    for key, value in experiment_config.items():
        setattr(args, key, value)
    args.write_tensorboard = False
    args.inference_config = None

    # Merge data directories with base path
    args.output_path = args.base / args.experiment['output_path']
    args.output_path.mkdir(exist_ok=True, parents=True)
    base_output_path = deepcopy(args.output_path)
    save_yaml(experiment_config, args.output_path / 'experiment.yaml')

    args.model['path'] = args.base / args.model['path']
    if isinstance(args.testsets[args.experiment['testset']], (list, tuple)):
        args.testsets[args.experiment['testset']] = [args.base / p for p in args.testsets[args.experiment['testset']]]
    else:
        args.testsets[args.experiment['testset']] = args.base / args.testsets[args.experiment['testset']]


    results = {}
    def init_results(i, k, param):
        key, subkey = k.split('.')
        getattr(args, key)[subkey] = param

        if i == len(args.experiment['ablations']) -1:
            node = results
            for ak in args.experiment['ablations'].keys():
                akey, asubkey = ak.split('.')
                level = getattr(args, akey)[asubkey]
                node.setdefault(level, {})
                node = node[level]

    recursive_iterations(-1, args.experiment['ablations'], init_results)
    results_json_path = base_output_path / 'results.json'
    with open(results_json_path, 'w') as jf:
        json.dump(results, jf, indent=2)

    # define operation to be performed at each iteration
    def operation(i, k, param):
        # print(f'Level {i}: {k} with parameter {param}')
        key, subkey = k.split('.')
        getattr(args, key)[subkey] = param

        if i == len(args.experiment['ablations']) -1:

            # Set directories
            output_dir = ''
            for ak in args.experiment['ablations'].keys():
                akey, asubkey = ak.split('.')
                output_dir += f'/{getattr(args, akey)[asubkey]}'

            args.output_path = base_output_path / output_dir.strip('/')
            args.output_path.mkdir(exist_ok=True, parents=True)

            with open(args.output_path / 'metrics.csv', 'w') as f:
                f.write('file;metric;mean;std;printable\n\n')

            # run reconstruction
            # print(f'ARGS: {args}')
            ssim, psnr, nmse, time, mem, ser, ttv = reconstruct(args)
            
            # write results to disc
            with open(results_json_path, 'r') as jf:
                results = json.load(jf)

            node = results
            for level in output_dir.split('/')[1:]:
                node = node[level]
            if ssim: node.setdefault("ssim", ssim.astype(float)) 
            else: node.setdefault("ssim", "-")
            if psnr: node.setdefault("psnr", psnr.astype(float))
            else: node.setdefault("psnr", "-")
            if nmse: node.setdefault("nmse", nmse.astype(float))
            else: node.setdefault("nmse", "-")
            node.setdefault("time", time.astype(float))
            node.setdefault("memory", mem.astype(float))
            if ser: node.setdefault("ser", ser.astype(float))
            else: node.setdefault("ser", "-")
            if ttv: node.setdefault("ttv", ttv.astype(float))
            else: node.setdefault("ttv", "-")

            with open(results_json_path, 'w') as jf:
                json.dump(results, jf, indent=2)

    # iterate over ablations, create output directories, run reconstructions, store overall outputs
    recursive_iterations(-1, args.experiment['ablations'], operation)

    # Write cross-seed summary CSVs (aggregates results across seeds for each condition group)
    if 'data.seed' in args.experiment['ablations']:
        write_seed_summary(args)


if __name__ == "__main__":

    print(f'CUDA: {torch.cuda.is_available(), torch.cuda.device_count()}')
    main()
