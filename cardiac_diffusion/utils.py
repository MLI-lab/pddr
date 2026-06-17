import argparse
import os
import csv
import json
import subprocess
import torch
import imageio
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from itertools import combinations
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from scipy.interpolate import interp1d
from scipy.stats import ttest_rel

from cardiac_diffusion.metrics import compute_metrics
from cardiac_diffusion.metrics import ser as ser_metric
from cardiac_diffusion.mri import image2kspace


def compute_pairwise_method_statistics(
    metrics_csv_paths=None,
    method_names=None,
    output_csv_path=None,
    confidence_level=0.95,
    method_csv_groups=None,
):
    """
    Compute pairwise paired t-tests between methods from evaluate.py metrics CSV files.

    Each CSV is expected to follow the format written by save_metrics_csv in scripts/evaluate.py:
      - A sample/file identifier line (e.g. "P026_sax")
      - Metric lines like ";SSIM;mean;std;printable"
      - Repeated blocks for each sample and one final OVERALL block

    The test is paired across common sample identifiers.

    Args:
        metrics_csv_paths (list[str|Path]|None): At least two paths to metrics.csv files.
        method_names (list[str]|None): Optional method labels matching metrics_csv_paths order.
        output_csv_path (str|Path|None): Optional path to store pairwise test results.
        confidence_level (float): Confidence level for paired-difference interval (default: 0.95).
        method_csv_groups (dict[str, list[str|Path]]|None): Optional mapping from method name
            to multiple metrics.csv files (e.g., one per seed/run). If provided, each method's
            per-case metrics are averaged across runs before pairwise testing.

    Returns:
        dict: Nested results keyed by method pair and metric.
    """
    if method_csv_groups is not None:
        if len(method_csv_groups) < 2:
            raise ValueError("Need at least two methods in method_csv_groups for pairwise testing.")

        method_data = {}
        for method_name, csv_paths in method_csv_groups.items():
            if len(csv_paths) == 0:
                raise ValueError(f"No CSV files provided for method: {method_name}")
            method_data[str(method_name)] = average_case_metrics_over_runs(csv_paths)
    else:
        if metrics_csv_paths is None:
            raise ValueError("Either metrics_csv_paths or method_csv_groups must be provided.")

        if len(metrics_csv_paths) < 2:
            raise ValueError("Need at least two metrics.csv files for pairwise testing.")

        if method_names is not None and len(method_names) != len(metrics_csv_paths):
            raise ValueError("method_names must have the same length as metrics_csv_paths.")

        method_data = {}
        for i, csv_path in enumerate(metrics_csv_paths):
            csv_path = Path(csv_path)
            if not csv_path.exists():
                raise FileNotFoundError(f"Metrics CSV not found: {csv_path}")
            if method_names is None:
                method_name = _infer_method_name_from_path(csv_path)
            else:
                method_name = str(method_names[i])

            if method_name in method_data:
                raise ValueError(f"Duplicate method name detected: {method_name}")

            method_data[method_name] = _parse_metrics_csv_per_case(csv_path)

    return _compute_pairwise_from_method_data(
        method_data=method_data,
        output_csv_path=output_csv_path,
        confidence_level=confidence_level,
    )


def average_case_metrics_over_runs(metrics_csv_paths):
    """
    Average per-case metric values across multiple runs/seeds for a single method.

    Args:
        metrics_csv_paths (list[str|Path]): List of metrics.csv files for one method.

    Returns:
        dict: {case_id: {metric_name: averaged_value}}
    """
    runs = []
    for csv_path in metrics_csv_paths:
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"Metrics CSV not found: {csv_path}")
        runs.append(_parse_metrics_csv_per_case(csv_path))

    all_cases = sorted({case for run in runs for case in run.keys()})
    all_metrics = sorted({
        metric
        for run in runs
        for sample_metrics in run.values()
        for metric in sample_metrics.keys()
    })

    averaged = {}
    for case in all_cases:
        case_metrics = {}
        for metric in all_metrics:
            vals = []
            for run in runs:
                value = run.get(case, {}).get(metric)
                if value is None:
                    continue
                if np.isnan(value):
                    continue
                vals.append(value)
            if len(vals) > 0:
                case_metrics[metric] = float(np.mean(vals))

        if len(case_metrics) > 0:
            averaged[case] = case_metrics

    return averaged


def _compute_pairwise_from_method_data(method_data, output_csv_path=None, confidence_level=0.95):
    """Run pairwise paired t-tests from method->per-case metric maps."""

    all_metrics = sorted({
        metric
        for data in method_data.values()
        for sample_metrics in data.values()
        for metric in sample_metrics.keys()
    })

    pairwise_results = {}
    rows_for_csv = []

    for method_a, method_b in combinations(sorted(method_data.keys()), 2):
        key = f"{method_a}__vs__{method_b}"
        pairwise_results[key] = {}

        common_samples = sorted(
            set(method_data[method_a].keys()).intersection(set(method_data[method_b].keys()))
        )

        for metric in all_metrics:
            values_a = []
            values_b = []

            for sample in common_samples:
                metric_a = method_data[method_a][sample].get(metric)
                metric_b = method_data[method_b][sample].get(metric)
                if metric_a is None or metric_b is None:
                    continue
                if np.isnan(metric_a) or np.isnan(metric_b):
                    continue
                values_a.append(metric_a)
                values_b.append(metric_b)

            values_a = np.asarray(values_a, dtype=float)
            values_b = np.asarray(values_b, dtype=float)

            if values_a.size < 2:
                result = {
                    'n': int(values_a.size),
                    'mean_a': float(np.mean(values_a)) if values_a.size > 0 else np.nan,
                    'mean_b': float(np.mean(values_b)) if values_b.size > 0 else np.nan,
                    'mean_diff': float(np.mean(values_a - values_b)) if values_a.size > 0 else np.nan,
                    't_stat': np.nan,
                    'p_value': np.nan,
                    'cohen_dz': np.nan,
                    'ci_low': np.nan,
                    'ci_high': np.nan,
                }
            else:
                t_res = ttest_rel(values_a, values_b, nan_policy='omit')
                diff = values_a - values_b
                std_diff = np.std(diff, ddof=1)
                ci = t_res.confidence_interval(confidence_level=confidence_level)
                if std_diff == 0:
                    cohen_dz = np.nan
                else:
                    cohen_dz = float(np.mean(diff) / std_diff)

                result = {
                    'n': int(values_a.size),
                    'mean_a': float(np.mean(values_a)),
                    'mean_b': float(np.mean(values_b)),
                    'mean_diff': float(np.mean(diff)),
                    't_stat': float(t_res.statistic),
                    'p_value': float(t_res.pvalue),
                    'cohen_dz': cohen_dz,
                    'ci_low': float(ci.low),
                    'ci_high': float(ci.high),
                }

            pairwise_results[key][metric] = result
            rows_for_csv.append({
                'method_a': method_a,
                'method_b': method_b,
                'metric': metric,
                **result,
            })

    if output_csv_path is not None:
        _write_pairwise_stats_csv(rows_for_csv, output_csv_path)

    return pairwise_results


def _parse_metrics_csv_per_case(csv_path):
    """Parse per-case metric means from a metrics.csv generated by save_metrics_csv."""
    per_case = {}
    current_sample = None

    with open(csv_path, 'r') as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            # Sample identifier line: no ';' prefix
            if not line.startswith(';'):
                current_sample = line
                if current_sample == 'file' or current_sample == 'OVERALL':
                    current_sample = None
                    continue
                per_case.setdefault(current_sample, {})
                continue

            if current_sample is None:
                continue

            parts = line.split(';')
            # Expected format: ;METRIC;mean;std;printable
            if len(parts) < 3:
                continue

            metric_name = parts[1].strip()
            mean_str = parts[2].strip()
            if mean_str in {'', 'None', '-', 'nan', 'NaN'}:
                continue

            try:
                per_case[current_sample][metric_name] = float(mean_str)
            except ValueError:
                continue

    return per_case


def _infer_method_name_from_path(csv_path):
    """Infer method label from path by using the parent directory name."""
    parent_name = csv_path.parent.name
    return parent_name if parent_name else str(csv_path)


def _write_pairwise_stats_csv(rows, output_csv_path):
    """Write pairwise test results to a semicolon-separated CSV file."""
    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    headers = [
        'method_a',
        'method_b',
        'metric',
        'n',
        'mean_a',
        'mean_b',
        'mean_diff',
        't_stat',
        'p_value',
        'cohen_dz',
        'ci_low',
        'ci_high',
    ]

    with open(output_csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers, delimiter=';')
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def setup_latex_style():
    """
    Configure matplotlib rcParams to match LLNCS (Springer LNCS) paper style.
    Uses Computer Modern Roman (cmr10) shipped with matplotlib — no LaTeX install needed.
    All text is set to 8pt to match \\footnotesize in the LLNCS class.
    """
    matplotlib.rcParams.update({
        # Font family: Computer Modern Roman (serif)
        'font.family': 'serif',
        'font.serif': ['cmr10'],          # matplotlib's built-in CM Roman
        'mathtext.fontset': 'cm',         # CM math symbols for $N$, $Q$, etc.
        'axes.unicode_minus': False,      # fix minus sign with CM fonts

        # Font sizes
        'font.size': 8,
        'axes.labelsize': 8,
        'xtick.labelsize': 6,             # smaller tick labels to save space
        'ytick.labelsize': 6,
        'legend.fontsize': 6.5,           # compact legend

        # Line and frame aesthetics (thinner for small figures)
        'lines.linewidth': 1.0,
        'axes.linewidth': 0.5,
        'grid.linewidth': 0.4,
        'grid.alpha': 0.5,
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'xtick.major.size': 3,
        'ytick.major.size': 3,

        # Legend
        'legend.framealpha': 0.8,
        'legend.edgecolor': '0.8',
        'legend.borderpad': 0.3,
        'legend.handlelength': 1.5,
        'legend.handletextpad': 0.4,
        'legend.columnspacing': 1.0,

        # PDF backend: embed fonts, use Type 42 (TrueType) for quality
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'savefig.dpi': 300,
        'savefig.transparent': False,
    })


def normalize(x, batch_independent=False):
    """
    Normalize between 0 and 1. If batch_independent is True, normalize each image in the batch independently.
    """
    b, c, h, w = x.shape
    assert c ==1, "Only single-channel images are supported."

    if batch_independent:
        # Compute the minimum and maximum values for each image in the batch
        min_vals = x.view(x.size(0), -1).min(dim=1, keepdim=True)[0].view(x.size(0), 1, 1, 1)
        max_vals = x.view(x.size(0), -1).max(dim=1, keepdim=True)[0].view(x.size(0), 1, 1, 1)
    else:
        min_vals = x.min()
        max_vals = x.max()

    # Normalize each image independently
    x_normalized = (x - min_vals) / (max_vals - min_vals)

    return x_normalized

def tensor_to_gif(tensor, filename, duration=50):
    """
    Convert a tensor in NCHW format to a GIF and save it.

    Args:
        tensor (torch.Tensor): The input tensor in NCHW format.
        filename (str): The filename to save the GIF.
        duration (float): The duration for each frame in the GIF (ms).
    """
    # Ensure the tensor is on the CPU and convert to numpy
    tensor = tensor.cpu().numpy()

    # # Clip high and low values: already done
    # tensor = np.clip(tensor, a_min=tensor.min(), a_max=np.percentile(tensor, 99.95))

    # Normalize the tensor to the range [0, 255]
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min()) * 255
    tensor = tensor.astype('uint8')

    # Convert the tensor to a list of PIL images
    images = []
    for i in range(tensor.shape[0]):
        frame = tensor[i]
        if frame.shape[0] == 1:  # Grayscale
            frame = frame.squeeze(0)
            image = Image.fromarray(frame, mode='L')
        elif frame.shape[0] == 3:  # RGB
            frame = frame.transpose(1, 2, 0)
            image = Image.fromarray(frame, mode='RGB')
        else:
            raise ValueError("Unsupported number of channels")
        images.append(image)

    # Save the images as a GIF
    imageio.mimsave(filename, images, duration=duration, loop=None) # loop=0 for looping

def save_tensors(
        path,
        init,
        ground,
        reconstruction,
        mask,
        fname
    ):
    # define output directory
    output_dir = path / 'output'
    output_dir.mkdir(exist_ok=True)

    # detach every tensor
    init = init.detach()
    ground = ground.detach() if ground is not None else None
    reconstruction = reconstruction.detach()
    mask = mask.detach()
    fname = fname[0] if fname is not None else None

    # correct shape and type
    init = init.squeeze(0)
    init = torch.complex(init[0], init[1])
    ground = ground.squeeze(0) if ground is not None else None
    reconstruction = reconstruction.squeeze(0)
    reconstruction = torch.complex(reconstruction[0], reconstruction[1])
    mask = mask.squeeze()

    # print(f'init: {init.shape}, type: {init.dtype}')
    # print(f'ground: {ground.shape}, type: {ground.dtype}')
    # print(f'reconstruction: {reconstruction.shape}, type: {reconstruction.dtype}')
    # print(f'mask: {mask.shape}, type: {mask.dtype}')

    # save file name
    if fname is not None:
        with open(output_dir / 'filename.txt', 'w') as f:
            f.write(f'{fname}\n')

    # save tensors 
    torch.save(init, output_dir / 'init.pt')
    if ground is not None:
        torch.save(ground, output_dir / 'ground.pt')
    torch.save(reconstruction, output_dir / 'reconstruction.pt')
    torch.save(mask, output_dir / 'mask.pt')

def save_image(tensor, filename):
    """
    Save a tensor in CHW format as an image.
    
    Args:
        tensor (torch.Tensor): The input tensor in CHW format.
        filename (str): The filename to save the image.
    """
    # Ensure the tensor is on the CPU and convert to numpy
    tensor = tensor.cpu().numpy()

    # Normalize the tensor to the range [0, 255]
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min()) * 255
    tensor = tensor.astype('uint8')

    # Convert the tensor to a PIL image
    if tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
        image = Image.fromarray(tensor, mode='L')
    elif tensor.shape[0] == 3:
        tensor = tensor.transpose(1, 2, 0)
        image = Image.fromarray(tensor, mode='RGB')
    else:
        raise ValueError("Unsupported number of channels")
    
    # Save the image
    image.save(filename)

def plot_images(directory, index=0, save=False, add_time_profiles=True, add_title=False):
    """
    Load tensors from the specified directory, compute image metrics, and plot the images.

    Args:
        directory (str): The directory path containing the tensors.
        index (int): The index of the frame to plot along the first dimension (default is 0).
    """
    # Load tensors
    init = torch.load(os.path.join(directory, 'init.pt'))
    ground = torch.load(os.path.join(directory, 'ground.pt'))
    reconstruction = torch.load(os.path.join(directory, 'reconstruction.pt'))
    mask = torch.load(os.path.join(directory, 'mask.pt'))
    lps = torch.load(os.path.join(directory, 'lps.pt'))

    # Ensure the tensors are on the CPU
    init = init.cpu()
    ground = ground.cpu()
    reconstruction = reconstruction.cpu()
    mask = mask.cpu()
    lps = lps.cpu()

    # Take the absolute value of the complex tensors
    init = init.abs()
    ground = ground.abs()
    reconstruction = reconstruction.abs()
    lps = lps.abs()

    # Select the specified frame
    if mask.ndim == 3:
        mask_frame = mask[index]
        mask_temp = mask[:, :, mask.shape[2] // 2]
    else:
        mask_frame = mask
        mask_temp = None

    ## optional crop TODO: do a center crop to a fixed image size
    print(ground.shape)
    gf, gx, gy = ground.shape
    left, right, up, down = 0, 0, 0, 0 # 80, 40, 0, 60
    ground = ground[:, up:gx-down, left:gy-right]
    init = init[:, up:gx-down, left:gy-right]
    reconstruction = reconstruction[:, up:gx-down, left:gy-right]
    mask_frame = mask_frame[up:gx-down, left:gy-right]
    lps = lps[:, up:gx-down, left:gy-right]
    if mask_temp is not None: mask_temp = mask_temp[:, up:gx-down]
    print(ground.shape)
    ##

    # reconstruction = np.clip(reconstruction, a_min=reconstruction.min(), a_max=np.percentile(reconstruction, 99.95))
    # Compute metrics
    _, _, _, metric_recon = compute_metrics(ground.unsqueeze(1), reconstruction.unsqueeze(1), return_image=True)
    _, _, _, metric_init = compute_metrics(ground.unsqueeze(1), init.unsqueeze(1), return_image=True)
    _, _, _, metric_lps = compute_metrics(ground.unsqueeze(1), lps.unsqueeze(1), return_image=True)

    # metric_recon = np.clip(metric_recon, a_min=metric_recon.min(), a_max=np.percentile(metric_recon, 99.95))
    
    # Plot and optionally save the images
    def plot_and_save(image, title, filename):
        plt.figure()
        # print(image.min(), image.max())
        # print(image.min(), image.max())
        plt.imshow(image, cmap='gray')
        if add_title : plt.title(title) 
        plt.axis('off')
        if save:
            plt.savefig(os.path.join(directory, filename), bbox_inches='tight', pad_inches=0)
        plt.show()

    # Plot ground truth
    gf, gy, gx = ground.shape
    ground_frame = ground[index]
    plot_and_save(ground_frame.numpy(), 'Ground Truth', 'ground_truth.png')
    if add_time_profiles:
        plot_and_save(ground[:, gy//2].numpy(), 'Ground Time profile xt', 'ground_xt.png')
        plot_and_save(ground[:, :, gx//2].numpy(), 'Ground Time profile yt', 'ground_yt.png')

    # Plot mask
    plot_and_save(mask_frame.numpy(), 'Mask', 'mask.png')
    if add_time_profiles and mask_temp is not None:
        plot_and_save(mask_temp.numpy(), 'Mask Temporal', 'mask_temp.png')

    # Plot init with metrics
    init_metric_image = metric_init[index]
    plot_and_save(init_metric_image.permute(1, 2, 0).numpy(), 'Init', 'init.png')
    if add_time_profiles:
        plot_and_save(init[:, gy//2].numpy(), 'Init Time profile xt', 'init_xt.png')
        plot_and_save(init[:, :, gx//2].numpy(), 'Init Time profile yt', 'init_yt.png')

    # Plot reconstruction with metrics
    reconstruction_metric_image = metric_recon[index]
    plot_and_save(reconstruction_metric_image.permute(1, 2, 0).numpy(), 'Reconstruction', 'reconstruction.png')
    if add_time_profiles:
        plot_and_save(reconstruction[:, gy//2].numpy(), 'Reconstruction Time profile xt', 'reconstruction_xt.png')
        plot_and_save(reconstruction[:, :, gx//2].numpy(), 'Reconstruction Time profile yt', 'reconstruction_yt.png')

    # Plot lps with metrics
    lps_metric_image = metric_lps[index]
    plot_and_save(lps_metric_image.permute(1, 2, 0).numpy(), 'L+S', 'lps.png')
    if add_time_profiles:
        plot_and_save(lps[:, gy//2].numpy(), 'L+S Time profile xt', 'lps_xt.png')
        plot_and_save(lps[:, :, gx//2].numpy(), 'L+S Time profile yt', 'lps_yt.png')


def plot_image_from_tensor(tensor, index=0, add_time_profiles=True, add_title=False, create_gif=True, save=True, crop_lrud=(48,4,0,17), cut_xy=(60,126)):

    image = torch.load(tensor, map_location='cpu')

    image = image.cpu()

    print(image.shape)
    if image.ndim == 5:
        image = image.squeeze()  # Remove batch dimension if present
        image = torch.complex(image[0], image[1])  # Convert to complex if needed

    image = image.abs()
    image = np.clip(image, a_min=image.min(), a_max=np.percentile(image, 99.9))
    if create_gif and save:
        tensor_path = Path(tensor)
        tensor_to_gif(image.unsqueeze(1), tensor_path.parent / f"{tensor_path.stem}.gif", duration=50)


    # optional crop TODO: do a center crop to a fixed image size
    print(image.shape)
    gf, gy, gx = image.shape
    image = image[:, crop_lrud[2]:gy-crop_lrud[3], crop_lrud[0]:gx-crop_lrud[1]]

    print(image.shape)
    gf, gy, gx = image.shape
    image_frame = image[index]

    y_cut =  cut_xy[0] # gy // 2
    x_cut =  cut_xy[1] # gx // 2
    print(y_cut, x_cut)

    # Plot and optionally save the images
    def plot_and_save(image, title, filename, show_cut=False):
        plt.figure()
        plt.imshow(image, cmap='gray')
        if add_title : plt.title(title) 
        plt.axis('off')
        if show_cut:
            plt.axvline(x=x_cut, color='red', alpha=0.25, ls='--')
            plt.axhline(y=y_cut, color='red', alpha=0.25, ls='--')
        if save:
            plt.savefig(tensor_path.parent / f"{filename}.png", bbox_inches='tight', pad_inches=0)
            plt.close()
        plt.show()

    plot_and_save(image_frame.numpy(), 'Reconstruction', 'reconstruction')
    plot_and_save(image_frame.numpy(), 'Reconstruction', 'reconstruction_cut', show_cut=True)
    if add_time_profiles:
        plot_and_save(image[:, y_cut].numpy(), 'Reconstruction Time profile xt', 'reconstruction_xt')
        plot_and_save(image[:, :, x_cut].numpy(), 'Reconstruction Time profile yt', 'reconstruction_yt')


def _load_tensor(tensor_path):
    """Load a .pt tensor and convert to a real-valued magnitude array (F, H, W)."""
    image = torch.load(tensor_path, map_location='cpu').cpu()
    if image.ndim == 5:
        image = image.squeeze()
        image = torch.complex(image[0], image[1])
    if image.is_complex():
        image = image.abs()
    else:
        image = image.abs()
    # Squeeze singleton channel dim: (F, 1, H, W) -> (F, H, W)
    if image.ndim == 4 and image.shape[1] == 1:
        image = image.squeeze(1)
    return image


def _save_video_ffmpeg(frames, output_path, fps=20, quality=2, loop_count=1):
    """Save a list of RGB numpy frames as a video using ffmpeg directly.

    Encodes with MPEG-4 Part 2 (DivX-compatible) in an AVI container.
    All metadata is stripped to avoid identifying information.

    Args:
        frames: list of numpy uint8 arrays, shape (H, W, 3).
        output_path: Output file path (should end in .avi).
        fps: Frames per second.
        quality: ffmpeg quality parameter (1-31, lower = better).
        loop_count: Number of times to repeat the frame sequence (default 1).
    """
    h, w = frames[0].shape[:2]
    looped_frames = frames * loop_count
    raw_data = b''.join(np.ascontiguousarray(f).tobytes() for f in looped_frames)

    cmd = [
        'ffmpeg', '-y',
        # Input: raw RGB frames piped via stdin
        '-f', 'rawvideo',
        '-pix_fmt', 'rgb24',
        '-s', f'{w}x{h}',
        '-r', str(fps),
        '-i', 'pipe:0',
        # Strip all metadata (no author, encoder, creation-time, etc.)
        '-map_metadata', '-1',
        '-fflags', '+bitexact',
        '-flags:v', '+bitexact',
        # MPEG-4 Part 2 codec (DivX-compatible)
        '-c:v', 'mpeg4',
        '-q:v', str(quality),
        output_path,
    ]

    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    _, stderr = proc.communicate(input=raw_data)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed (exit {proc.returncode}):\n{stderr.decode()[-500:]}"
        )


def create_supplementary_video(
    tensor_label_pairs,
    save_path,
    ground=None,
    duration=50,
    fps=20,
    save_mp4=False,
    crop=None,
    padding=12,
    label_height=30,
    loop_count_mp4=1
):
    """
    Create a comparison video/GIF with multiple reconstruction methods shown
    in a 2-row grid on a black background with white method labels.

    When *ground* is provided (path to a ground-truth .pt tensor), image-quality
    metrics (SSIM, PSNR) are computed per frame and overlaid on every
    reconstruction panel, while the ground truth itself is shown as-is in the
    first panel.

    Args:
        tensor_label_pairs (list of tuples): Each element is (tensor_path, label).
            The .pt file should contain a tensor of shape (F, H, W) or (1, 2, F, H, W).
        save_path (str or Path): Directory where the output files are saved.
        ground (str or Path, optional): Path to a ground-truth .pt tensor.  When
            given, the ground truth is shown in the first panel and metric images
            are used for every other panel.
        duration (float): Frame duration in ms for the GIF (default 50 ms = 20 fps).
        fps (int): Frames per second for the video output.
        save_mp4 (bool): If True, also save an AVI video (MPEG-4 Part 2 codec,
            metadata-free) alongside the GIF.
        crop (tuple, optional): (top, bottom, left, right) pixels to crop from each
            edge of every frame. If None, no cropping is applied.
        padding (int): Pixel spacing between panels and around the border.
        label_height (int): Vertical space reserved above each panel for the label.
        loop_count_mp4 (int): Number of times to repeat the frame sequence in the
            MP4 video (1 for no looping). Default is 1.
    """
    save_path = Path(save_path)

    # Determine output paths: if save_path ends with .gif/.mp4/.avi, treat it as
    # an explicit filename; otherwise it's a directory and we pick default names.
    if save_path.suffix.lower() in ('.gif', '.mp4', '.avi'):
        out_dir = save_path.parent
        gif_path = save_path.with_suffix('.gif')
        video_path = save_path.with_suffix('.avi')
    else:
        out_dir = save_path
        gif_path = save_path / 'supplementary_video.gif'
        video_path = save_path / 'supplementary_video.avi'
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load a bold font (try system TTF bold faces, fall back to PIL default) ---
    font = None
    font_size = max(18, label_height - 8)
    for font_path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-Bold.ttf",
    ]:
        if os.path.isfile(font_path):
            font = ImageFont.truetype(font_path, font_size)
            break
    if font is None:
        font = ImageFont.load_default(size=font_size)

    # --- Optionally load ground truth ---
    ground_tensor = None
    if ground is not None:
        ground_tensor = _load_tensor(ground)  # (F, H, W) torch tensor

    # --- Load & preprocess every tensor ---
    # Each entry in *panels* is  (label, frames_array, mean_metrics)
    #   frames_array: numpy uint8, shape (F, H, W) for grayscale
    #                                 or (F, H, W, 3) for RGB metric images
    #   mean_metrics: None or dict with 'ssim' and 'psnr' mean values
    panels = []

    for tensor_path, label in tensor_label_pairs:
        image = _load_tensor(tensor_path)  # (F, H, W) torch tensor
        image_clipped = torch.tensor(
            np.clip(image.numpy(), a_min=image.min().item(), a_max=np.percentile(image.numpy(), 99.9))
        )

        # Optional spatial crop (applied before metric computation so sizes match ground truth)
        if crop is not None:
            top, bottom, left, right = crop
            _, h, w = image_clipped.shape
            image_clipped = image_clipped[:, top:h - bottom, left:w - right]

        if ground_tensor is not None:
            # Prepare ground truth with matching crop
            gt = torch.tensor(
                np.clip(ground_tensor.numpy(), a_min=ground_tensor.min().item(),
                        a_max=np.percentile(ground_tensor.numpy(), 99.9))
            )
            if crop is not None:
                top, bottom, left, right = crop
                _, h, w = gt.shape
                gt = gt[:, top:h - bottom, left:w - right]

            # compute_metrics expects (B, C, H, W); treat frames as batch, 1 channel
            ssims, psnrs, _, metric_images = compute_metrics(
                gt.unsqueeze(1), image_clipped.unsqueeze(1), return_image=True
            )
            mean_metrics = {
                'ssim': float(np.mean(ssims)),
                'psnr': float(np.mean(psnrs)),
            }
            # metric_images: (F, C, H, W) — C may be 1 (grayscale) or 3 (RGB)
            metric_images = metric_images.permute(0, 2, 3, 1).contiguous()  # (F, H, W, C)
            if metric_images.shape[-1] == 1:
                metric_images = metric_images.expand(-1, -1, -1, 3).contiguous()
            frames_rgb = (metric_images.numpy() * 255).astype(np.uint8)  # (F, H, W, 3)

            # Draw mean metrics on each frame, below the per-frame metrics
            # (compute_metrics draws SSIM at y=0, PSNR at y=18, font size=18)
            inframe_font = ImageFont.load_default(size=18)
            for fi in range(frames_rgb.shape[0]):
                frame_pil = Image.fromarray(frames_rgb[fi], mode='RGB')
                frame_draw = ImageDraw.Draw(frame_pil)
                frame_draw.text((0, 54), f"SSIM={mean_metrics['ssim']*100:.1f}%", fill='yellow', font=inframe_font)
                frame_draw.text((0, 72), f"PSNR={mean_metrics['psnr']:.1f}dB", fill='yellow', font=inframe_font)
                frames_rgb[fi] = np.array(frame_pil)

            panels.append((label, frames_rgb, mean_metrics))
        else:
            # No ground truth → plain grayscale panels
            image_np = image_clipped.numpy()
            mn, mx = image_np.min(), image_np.max()
            if mx - mn > 0:
                image_np = (image_np - mn) / (mx - mn) * 255.0
            panels.append((label, image_np.astype(np.uint8), None))

    # Prepend ground truth panel (shown as-is, grayscale) when available
    if ground_tensor is not None:
        gt_np = ground_tensor.numpy()
        gt_np = np.clip(gt_np, a_min=gt_np.min(), a_max=np.percentile(gt_np, 99.9))
        if crop is not None:
            top, bottom, left, right = crop
            _, h, w = gt_np.shape
            gt_np = gt_np[:, top:h - bottom, left:w - right]
        mn, mx = gt_np.min(), gt_np.max()
        if mx - mn > 0:
            gt_np = (gt_np - mn) / (mx - mn) * 255.0
        panels.insert(0, ('Ground Truth', gt_np.astype(np.uint8), None))

    num_panels = len(panels)
    num_frames = min(p[1].shape[0] for p in panels)

    # Grid layout: 2 rows, ceil(N/2) columns
    ncols = int(np.ceil(num_panels / 2))
    nrows = 2 if num_panels > 1 else 1

    # Determine the maximum panel size (all panels use the same cell size)
    max_h = max(p[1].shape[1] for p in panels)
    max_w = max(p[1].shape[2] for p in panels)

    canvas_w = ncols * max_w + (ncols + 1) * padding
    canvas_h = nrows * (max_h + label_height) + (nrows + 1) * padding

    # --- Compose frames ---
    frames = []
    for f_idx in range(num_frames):
        canvas = Image.new('RGB', (canvas_w, canvas_h), color=(0, 0, 0))
        draw = ImageDraw.Draw(canvas)

        for idx, (label, img, mean_metrics) in enumerate(panels):
            row = idx // ncols
            col = idx % ncols

            # Top-left corner of this cell's image area
            x0 = (col + 1) * padding + col * max_w
            y0 = (row + 1) * padding + row * (max_h + label_height) + label_height

            # Centre the panel if it's smaller than the max size
            frame_h, frame_w = img.shape[1], img.shape[2]
            x_offset = (max_w - frame_w) // 2
            y_offset = (max_h - frame_h) // 2

            # Build PIL image (grayscale or RGB depending on whether metrics were computed)
            frame_data = np.ascontiguousarray(img[f_idx])
            if frame_data.ndim == 2:
                panel_img = Image.fromarray(frame_data, mode='L').convert('RGB')
            else:
                panel_img = Image.fromarray(frame_data, mode='RGB')
            canvas.paste(panel_img, (x0 + x_offset, y0 + y_offset))

            # Draw label centred above this panel
            bbox = draw.textbbox((0, 0), label, font=font)
            text_w = bbox[2] - bbox[0]
            text_x = x0 + (max_w - text_w) // 2
            text_y = y0 - label_height + (label_height - (bbox[3] - bbox[1])) // 2
            draw.text((text_x, text_y-6), label, fill=(255, 255, 255), font=font)

        frames.append(np.array(canvas))

    # --- Save outputs ---
    imageio.mimsave(str(gif_path), frames, duration=duration, loop=0)
    print(f"Saved GIF to {gif_path}")

    if save_mp4:
        _save_video_ffmpeg(frames, str(video_path), fps=fps, loop_count=loop_count_mp4)
        print(f"Saved video to {video_path}")


def compute_and_plot_ser(validation_data_path, reconstructions_path):
    def compute_ser(reconstruction, mask, measurements, sensitivities):
        assert reconstruction.ndim == 3, f"Expected reconstruction to have 3 dimensions (complex-valued: f, h, w), got {reconstruction.ndim}"

        masked_kspace = image2kspace(reconstruction.unsqueeze(1), sensitivities.unsqueeze(0), dim=(2, 3)) * mask
        masked_kspace = torch.stack((masked_kspace.real, masked_kspace.imag), dim=0)
        measurements = torch.stack((measurements.real, measurements.imag), dim=0)

        return ser_metric(measurements, masked_kspace, normalized=False)
    
    # load validation data and sensitivities
    validation_data_path = Path(validation_data_path)
    validation_mask = torch.load(validation_data_path / 'validation_mask.pt')
    measurements = torch.load(validation_data_path / 'validation_kspace.pt')
    sensitivities = torch.load(validation_data_path / 'sensitivities.pt')

    # gather all reconstruction files in directory
    reconstructions_path = Path(reconstructions_path)
    if reconstructions_path.is_dir():
        reconstructions = sorted(reconstructions_path.glob(f'*reconstruction_*.pt'), key=lambda x: int(x.stem.split('_')[-1]))
    elif reconstructions_path.is_file():
        reconstructions = [reconstructions_path]
    else:
        raise ValueError(f"{reconstructions_path} is not a valid directory or file.")
    
    # iterate over all reconstructions
    sers = []
    for recon in reconstructions:
        print(f"Processing {recon}...")
        reconstruction = torch.load(recon).cpu()

        ser = compute_ser(reconstruction, validation_mask, measurements, sensitivities)
        print(f"SER for {recon.name}: {ser:.2f} dB")
        sers.append((int(recon.stem.split('_')[-1]), ser))

    # plot curve and save values to make a plot of all methods together
    x, y = map(list, zip(*sers))
    plt.plot(x, y)
    plt.show()

    with open(reconstructions_path / 'ser.json', 'w') as file:
        json.dump(sers, file)


def _plot_on_axis(ax, json_paths_and_labels, styles, xlabel, ylabel):
    """
    Plot random / sliding-window PSNR curves on a given matplotlib Axes.
    Shared helper used by both plot_comparison and plot_comparison_side_by_side.

    Returns:
        handles (list): Line2D handles for legend construction.
    """
    if styles is None:
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
                  'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
        styles = [(1.0, colors[i], '-') for i in range(len(json_paths_and_labels))]

    handles = []
    for (json_path, method_label), (alpha, color, linestyle) in zip(json_paths_and_labels, styles):
        with open(json_path, 'r') as f:
            values = json.load(f)

        x = [12, 24, 36, 48, 60, 72, 84, 96, 108, 120]
        y = [values['random'][k] for k in values['random'].keys()]
        y = [d['psnr'] if 'psnr' in d else None for d in y]
        rh, = ax.plot(x, y, label=f'Random: $K$={method_label}',
                      alpha=alpha, color='tab:orange', linestyle=linestyle)

        y = [values['sliding_window'][k] for k in values['sliding_window'].keys()]
        y = [d['psnr'] if 'psnr' in d else None for d in y]
        swh, = ax.plot(x, y, label=f'Sliding window: $K$={method_label}',
                       alpha=alpha, color='tab:green', linestyle=linestyle)

        handles += [rh, swh]

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(12, 121, 12))     # ticks every 20, matching the original plots
    ax.grid(True)
    return handles


def _plot_multi_series_on_axis(ax, json_path, series, xlabel, ylabel, metric='psnr',
                               mark_max=False):
    """
    Plot multiple series from a single JSON file on a matplotlib Axes.

    The JSON has top-level keys for each series, and fractional sub-keys
    (e.g. "0.1" through "1.0") each containing a metric dict.

    Args:
        ax:        matplotlib Axes to plot on.
        json_path: Path to the JSON file.
        series:    List of (json_key, label, color) tuples defining each line.
        xlabel, ylabel: Axis labels.
        metric:    Which metric to extract from each sub-dict (default 'psnr').
        mark_max:  If True, mark the maximum value with a star marker.

    Returns:
        handles: List of Line2D handles for legend construction.
    """
    with open(json_path, 'r') as f:
        values = json.load(f)

    handles = []
    for json_key, label, color in series:
        sub = values[json_key]
        x_all = [int(round(float(k) * 1000)) for k in sub.keys()]
        y_all = []
        for k in sub:
            d = sub[k]
            if isinstance(d, dict) and metric in d:
                y_all.append(d[metric])
            else:
                y_all.append(None)
        # Trim trailing None values (e.g. empty dicts)
        while y_all and y_all[-1] is None:
            y_all.pop()
        x_trimmed = x_all[:len(y_all)]
        h, = ax.plot(x_trimmed, y_all, label=label, color=color)
        if mark_max:
            valid = [(x, y) for x, y in zip(x_trimmed, y_all) if y is not None]
            if valid:
                best_x, best_y = max(valid, key=lambda p: p[1])
                ax.scatter(best_x, best_y, color=color, marker='*', s=30, zorder=5)
        handles.append(h)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(200, 1001, 200))
    ax.grid(True)
    return handles


def _plot_dict_series_on_axis(ax, json_paths_and_labels, sub_key, metric, xlabel, ylabel,
                              transform=None):
    """
    Plot one line per JSON file on a matplotlib Axes.

    Each JSON has top-level keys that are x-values (e.g. block sizes "12", "24", ...)
    and nested sub-keys (e.g. "0.05") containing metric dicts.

    Args:
        ax:                   matplotlib Axes.
        json_paths_and_labels: list of (json_path, label, color) tuples.
        sub_key:              Which nested key to select (e.g. '0.05').
        metric:               Which metric to read (e.g. 'psnr', 'memory').
        xlabel, ylabel:       Axis labels.
        transform:            Optional callable applied to each y value (e.g. lambda v: v/1024).

    Returns:
        handles: List of Line2D handles.
    """
    colors_default = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
    handles = []
    all_x = set()
    for i, item in enumerate(json_paths_and_labels):
        json_path, label = item[0], item[1]
        color = item[2] if len(item) > 2 else colors_default[i]
        with open(json_path, 'r') as f:
            values = json.load(f)

        x_vals, y_vals = [], []
        for k in values:
            sub = values[k].get(sub_key, {})
            val = sub.get(metric) if isinstance(sub, dict) else None
            if val is not None:
                if transform is not None:
                    val = transform(val)
                x_vals.append(int(k))
                y_vals.append(val)
                all_x.add(int(k))
        h, = ax.plot(x_vals, y_vals, label=label, color=color)
        handles.append(h)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if all_x:
        step = min(diff for a, b in zip(sorted(all_x), sorted(all_x)[1:]) if (diff := b - a) > 0) if len(all_x) > 1 else 12
        ax.set_xticks(sorted(all_x))
    ax.grid(True)
    return handles


def _plot_tradeoff_scatter_on_axis(
    ax,
    points,
    x_key,
    xlabel,
    ylabel='PSNR (dB)',
    color_by='k',
    marker_by='q',
    color_map=None,
    marker_map=None,
    marker_size=30,
    fit_curve=False,
):
    """
    Plot a point-only tradeoff scatter on a matplotlib Axes with interpolated curves.

    Args:
        ax: matplotlib Axes.
        points: List of dicts containing at least x_key, 'psnr', color_by, marker_by.
        x_key: Which key to use on x-axis (e.g. 'time' or 'vram').
        xlabel, ylabel: Axis labels.
        color_by: Key used to group/color points (default: 'k').
        marker_by: Key used to pick marker shape (default: 'q').
        color_map: Optional dict mapping group values to matplotlib colors.
        marker_map: Optional dict mapping marker group values to marker styles.
        marker_size: Scatter marker size.
        fit_curve: If True, fit quadratic spline curves through each K-group (default True).

    Returns:
        handles: List of handles for legend construction (lines for K, markers for Q).
    """
    if color_map is None:
        color_map = {
            100: 'tab:blue',
            200: 'tab:orange',
            400: 'tab:green',
        }

    if marker_map is None:
        marker_map = {
            12: 'o',
            36: '*',
            120: 's',
        }

    groups = sorted({p[color_by] for p in points})
    marker_groups = sorted({p[marker_by] for p in points})
    handles = []
    for g in groups:
        # Create one legend handle per K color (as a line for the curve).
        h_line = ax.plot([], [], color=color_map.get(g, 'tab:gray'),
                        linestyle='-', linewidth=1.5, label=fr'$K$={g}', alpha=0.95)[0]
        handles.append(h_line)

        # Interpolate quadratic spline through all points in this K group (interpolation only, no extrapolation).
        if fit_curve:
            group_all = [p for p in points if p[color_by] == g]
            xs_all = np.array([p[x_key] for p in group_all])
            ys_all = np.array([p['psnr'] for p in group_all])
            if len(xs_all) >= 3:  # Need at least 3 points for quadratic interpolation
                # Sort by x for monotonic input
                sort_idx = np.argsort(xs_all)
                xs_sorted = xs_all[sort_idx]
                ys_sorted = ys_all[sort_idx]
                try:
                    # Quadratic interpolation—3 points uniquely define a parabola
                    # Use bounds_error=True and no fill_value to strictly interpolate within data range
                    interp_func = interp1d(xs_sorted, ys_sorted, kind='quadratic', bounds_error=True)
                    # Generate smooth curve only over the data range (no extrapolation)
                    x_smooth = np.linspace(xs_sorted.min(), xs_sorted.max(), 150)
                    y_smooth = interp_func(x_smooth)
                    ax.plot(x_smooth, y_smooth, color=color_map.get(g, 'tab:gray'),
                           linestyle='-', linewidth=1.5, alpha=0.35)
                except Exception as e:
                    pass  # If interpolation fails, just skip the curve

        for q in marker_groups:
            subset = [p for p in points if p[color_by] == g and p[marker_by] == q]
            if not subset:
                continue
            xs = [p[x_key] for p in subset]
            ys = [p['psnr'] for p in subset]
            ax.scatter(
                xs,
                ys,
                s=marker_size,
                color=color_map.get(g, 'tab:gray'),
                marker=marker_map.get(q, 'o'),
                alpha=0.95,
            )

    # Add marker-style legend entries for Q (neutral color to avoid color conflict with K).
    for q in marker_groups:
        hq = ax.scatter([], [], s=marker_size, color='black', marker=marker_map.get(q, 'o'),
                        label=fr'$Q$={q}', alpha=0.95)
        handles.append(hq)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)
    return handles


def plot_comparison(json_paths_and_labels, styles=None, xlabel=None, ylabel=None, save_path=None):
    """
    Plot curves for multiple methods from their respective .json files.

    Args:
        json_paths_and_labels (list of tuples): List of (json_path, method_label).
        save_path (str, optional): If provided, saves the plot to this path.
    """
    if styles is None:
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
        styles = [(1.0, colors[i], '-') for i in range(len(json_paths_and_labels))]

    plt.figure(figsize=(8, 4)) # (8, 4)
    handles = []
    for (json_path, method_label), (alpha, color, linestyle) in zip(json_paths_and_labels, styles):
        with open(json_path, 'r') as f:
            values = json.load(f)

        # # for SER jsons:
        # x, y = map(list, zip(*values))

        # # for dict jsons:
        # x = [int(v) for v in list(values.keys())]
        # y = [values[k]['0.05'] for k in values.keys()]

        # y = [d['psnr'] if 'psnr' in d else None for d in y]
        # # y = [d['memory'] / 1024 if 'memory' in d else None for d in y]

        # # plot curve
        # h, = plt.plot(x, y, label=method_label, alpha=alpha, color=color, linestyle=linestyle)
        # handles.append(h)

        x = [12, 24, 36, 48, 60, 72, 84, 96, 108, 120]
        y = [values['random'][k] for k in values['random'].keys()]
        y = [d['psnr'] if 'psnr' in d else None for d in y]
        rh, = plt.plot(x, y, label=f'Random: $K$={method_label}', alpha=alpha, color='tab:orange', linestyle=linestyle)

        y = [values['sliding_window'][k] for k in values['sliding_window'].keys()]
        y = [d['psnr'] if 'psnr' in d else None for d in y]
        swh, = plt.plot(x, y, label=f'Sliding window: $K$={method_label}', alpha=alpha, color='tab:green', linestyle=linestyle)

        handles += [rh, swh]

        # x = [100 * (k+1) for k in range(10)]
        # y = [values['4'][k] for k in values['4'].keys()]
        # y = [d['psnr'] if 'psnr' in d else None for d in y]
        # h1, = plt.plot(x, y, label='4x acceleration', alpha=alpha, color='tab:blue', linestyle=linestyle)
        # plt.scatter(x[2], y[2], color='tab:blue', marker='*')

        # y = [values['8'][k] for k in values['8'].keys()]
        # y = [d['psnr'] if 'psnr' in d else None for d in y]
        # h2, = plt.plot(x, y, label='8x acceleration', alpha=alpha, color='tab:orange', linestyle=linestyle)
        # plt.scatter(x[3], y[3], color='tab:orange', marker='*')

        # y = [values['12'][k] for k in values['12'].keys()]
        # y = [d['psnr'] if 'psnr' in d else None for d in y]
        # h3, = plt.plot(x, y, label='12x acceleration', alpha=alpha, color='tab:green', linestyle=linestyle)
        # plt.scatter(x[4], y[4], color='tab:green', marker='*')


        # x = [100 * (k+1) for k in range(10)]

        # y = [values['descending'][k] for k in values['descending'].keys()]
        # y = [d['psnr'] if 'psnr' in d else None for d in y]
        # h2, = plt.plot(x, y, label='descending', alpha=alpha, color='tab:green', linestyle=linestyle)
        # # plt.scatter(x[3], y[3], color='tab:orange', marker='*')

        # y = [values['random'][k] for k in values['random'].keys()]
        # y = [d['psnr'] if 'psnr' in d else None for d in y]
        # h3, = plt.plot(x, y, label='random', alpha=alpha, color='tab:orange', linestyle=linestyle)
        # # plt.scatter(x[4], y[4], color='tab:green', marker='*')
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # plt.title('')
    plt.legend(handles=[handles[0], handles[2], handles[4], handles[1], handles[3], handles[5]], loc='best') 
    # plt.legend(loc='lower right') # 'lower right' # 'best'
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()


def plot_comparison_side_by_side(
    left_data,
    right_data,
    left_xlabel,
    right_xlabel,
    left_ylabel,
    right_ylabel,
    left_styles=None,
    right_styles=None,
    save_path=None,
    shared_legend=True,
    left_plot_fn=None,
    right_plot_fn=None,
    aspect_ratio=0.35,
):
    """
    Create a publication-ready side-by-side comparison figure (two subplots)
    with Computer Modern fonts matching Springer LNCS style.

    Produces a true vector PDF suitable for direct \\includegraphics insertion.
    The figure width matches the LLNCS textwidth of 12.2 cm.

    Args:
        left_data / right_data:   list of (json_path, label) tuples — same format as plot_comparison.
        left_xlabel / right_xlabel: x-axis labels. Use $...$ for math italic (e.g. '$N$').
        left_ylabel / right_ylabel: y-axis labels.
        left_styles / right_styles: list of (alpha, color, linestyle) tuples, or None for defaults.
        save_path (str|Path):      Output file path. Use .pdf for vector PDF, .svg for SVG.
                                   If None, the figure is shown interactively.
        shared_legend (bool):      If True (default), place a single shared legend above both panels.
                                   If False, each subplot gets its own legend.
        left_plot_fn (callable):   Optional custom plotting function for the left panel.
                                   Signature: fn(ax) -> list_of_handles.
                                   When provided, left_data / left_styles are ignored.
        right_plot_fn (callable):  Same as left_plot_fn but for the right panel.
        aspect_ratio (float):      Height = width * aspect_ratio (default 0.35).
    """
    setup_latex_style()

    # LLNCS textwidth = 12.2 cm; convert to inches.  Use 2:1 aspect ratio
    # matching the original plot proportions (figsize 8×4).
    textwidth_inches = 15.25 / 2.54  # ≈ 4.803 in
    fig_height = textwidth_inches * aspect_ratio
    # textwidth_inches = 12.2 / 2.54  # ≈ 4.803 in
    # fig_height = textwidth_inches * 0.25  # relaxed ratio — gives plots more vertical room
    if shared_legend:
        fig_height += 1.0 # Add extra vertical space for the shared legend below the plots

    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=(textwidth_inches, fig_height),
    )

    # Plot left panel
    if left_plot_fn is not None:
        handles_left = left_plot_fn(ax1)
    else:
        handles_left = _plot_on_axis(ax1, left_data, left_styles, left_xlabel, left_ylabel)

    # Plot right panel
    if right_plot_fn is not None:
        handles_right = right_plot_fn(ax2)
    else:
        handles_right = _plot_on_axis(ax2, right_data, right_styles, right_xlabel, right_ylabel)

    # Optionally suppress duplicate y-label on right panel when labels match
    if left_ylabel == right_ylabel:
        ax2.set_ylabel('')

    _custom = left_plot_fn is not None or right_plot_fn is not None

    if shared_legend:
        if _custom:
            # Custom plot functions — combine handles from both panels
            all_handles = handles_left + handles_right
            # Deduplicate labels so shared legends stay compact for repeated series.
            unique_by_label = {}
            for h in all_handles:
                label = h.get_label()
                if label not in unique_by_label:
                    unique_by_label[label] = h
            legend_handles = list(unique_by_label.values())
            legend_labels = list(unique_by_label.keys())
            fig.legend(
                legend_handles, legend_labels,
                loc='lower center',
                ncol=max(min(len(legend_handles), 3), 1),
                bbox_to_anchor=(0.5, 0.0),
                frameon=True,
            )
        else:
            # Reorder handles so the 2×3 legend reads:
            #   Row 1: Random K=100    Random K=200    Random K=400
            #   Row 2: Sliding w. K=100  Sliding w. K=200  Sliding w. K=400
            # matplotlib fills ncol=3 column-wise, so we must interleave:
            #   [R100, SW100, R200, SW200, R400, SW400]
            random_handles = handles_left[0::2]    # [R100, R200, R400]
            sw_handles = handles_left[1::2]         # [SW100, SW200, SW400]
            ordered_handles = [h for pair in zip(random_handles, sw_handles) for h in pair]
            ordered_labels = [h.get_label() for h in ordered_handles]

            fig.legend(
                ordered_handles, ordered_labels,
                loc='lower center',
                ncol=3,
                bbox_to_anchor=(0.5, 0.0),
                frameon=True,
            )
        # Adjust spacing: room for legend below, plots fill the top
        fig.subplots_adjust(top=0.95, wspace=0.28, left=0.10, right=0.98, bottom=0.30)
    else:
        if _custom:
            # Custom plots: single legend on right subplot only (no duplicate legends).
            all_handles = handles_left + handles_right
            unique_by_label = {}
            for h in all_handles:
                label = h.get_label()
                if label not in unique_by_label:
                    unique_by_label[label] = h
            ax2.legend(handles=list(unique_by_label.values()), loc='best')
        else:
            # Single legend on the right subplot only (saves space, no duplication)
            n = len(handles_right)
            random_handles = handles_right[0::2]
            sw_handles = handles_right[1::2]
            ordered = random_handles + sw_handles
            ax2.legend(handles=ordered, loc='best')

    if save_path is not None:
        save_path = Path(save_path)
        fmt = save_path.suffix.lstrip('.')
        if fmt not in ('pdf', 'svg', 'eps'):
            fmt = 'pdf'
        fig.savefig(save_path, format=fmt, bbox_inches='tight', pad_inches=0.02)
        print(f'Saved vector figure to {save_path}')

    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--base", type=Path, required=True, metavar="/path/to/base_directory", help='Path to the base_directory.')
    args = parser.parse_args()

    # # --- Block sampling: side-by-side (sequence length + block size) ---
    plot_comparison_side_by_side(
        left_data=[
            (args.base / 'experiments/slab_sampling/sequence_size/100/results.json', '100'),
            (args.base / 'experiments/slab_sampling/sequence_size/200/results.json', '200'),
            (args.base / 'experiments/slab_sampling/sequence_size/400/results.json', '400'),
        ],
        right_data=[
            (args.base / 'experiments/slab_sampling/block_size/100/results.json', '100'),
            (args.base / 'experiments/slab_sampling/block_size/200/results.json', '200'),
            (args.base / 'experiments/slab_sampling/block_size/400/results.json', '400'),
        ],
        left_xlabel=r'Sequence length $N$',
        right_xlabel=r'Block size $Q$',
        left_ylabel='PSNR (dB)',
        right_ylabel='PSNR (dB)',
        left_styles=[(0.7, 'tab:green', '--'), (1.0, 'tab:green', '-'), (0.7, 'tab:green', ':')],
        right_styles=[(0.7, 'tab:green', '--'), (1.0, 'tab:green', '-'), (0.7, 'tab:green', ':')],
        save_path= args.base / 'experiments/block_sampling.pdf',
        shared_legend=False,
        aspect_ratio=0.25,
    )


    # # --- Create plots for prospective example ---
    plot_image_from_tensor(args.base / 'experiments/reconstruction_prospective_example/PDDR/pddr.pt', index=0, add_time_profiles=True, add_title=False, save=True)
    plot_image_from_tensor(args.base / 'experiments/reconstruction_prospective_example/FMLP/fmlp.pt', index=0, add_time_profiles=True, add_title=False, save=True)
    plot_image_from_tensor(args.base / 'experiments/reconstruction_prospective_example/TDIP/tdip.pt', index=0, add_time_profiles=True, add_title=False, save=True)
    plot_image_from_tensor(args.base / 'experiments/reconstruction_prospective_example/L+S/lps.pt', index=0, add_time_profiles=True, add_title=False, save=True)


    # # --- Create supplementary videos for prospective example ---
    create_supplementary_video(
        [(args.base / 'experiments/reconstruction_prospective_example/PDDR/pddr.pt', 'PDDR'),
        (args.base / 'experiments/reconstruction_prospective_example/L+S/lps.pt', 'L+S'),
        (args.base / 'experiments/reconstruction_prospective_example/FMLP/fmlp.pt', 'FMLP'),
        (args.base / 'experiments/reconstruction_prospective_example/TDIP/tdip.pt', 'T-DIP')],
        save_path=args.base / 'experiments/reconstruction_prospective_example',
        save_mp4=True,
        crop=(0, 0, 30, 0)
    )


    # # --- Create supplementary videos for retrospective examples ---
    for file in ['P026_5', 'P072_5', 'P097_4', 'P040_0', 'P040_1', 'P040_2']: 
        create_supplementary_video(
            [(args.base / f'experiments/reconstruction_retrospective_examples/PDDR/{file}/output/PDDR.pt', 'PDDR'),
            (args.base / f'experiments/reconstruction_retrospective_examples/DPS/{file}/output/DPS.pt', 'DPS'),
            (args.base / f'experiments/reconstruction_retrospective_examples/SDR/{file}/output/SDR.pt', 'SDR'),
            (args.base / f'experiments/reconstruction_retrospective_examples/ZF/{file}/output/ZF.pt', 'Zero-Filled'),
            (args.base / f'experiments/reconstruction_retrospective_examples/L+S/{file}/output/L+S.pt', 'L+S'),
            (args.base / f'experiments/reconstruction_retrospective_examples/FMLP/{file}/output/FMLP.pt', 'FMLP'),
            (args.base / f'experiments/reconstruction_retrospective_examples/TDIP/{file}/output/TDIP.pt', 'T-DIP')],
            save_path = args.base / f'experiments/reconstruction_retrospective_examples/videos/{file}.gif',
            ground = args.base / f'experiments/reconstruction_retrospective_examples/L+S/{file}/output/ground.pt',
            save_mp4=True,
            loop_count_mp4=10
        )


    # # --- Model ablation: side-by-side (PSNR + VRAM) ---
    # ma_base = args.base / 'experiments/model_ablation'
    # ma_models = [
    #     (f'{ma_base}/NEW_3D_UNetModel/results.json', '3D', 'tab:blue'),
    #     (f'{ma_base}/NEW_SpatioTemporalUNetModel_3Dtemporal/results.json', '2D/3D', 'tab:orange'),
    #     (f'{ma_base}/NEW_SpatioTemporalUNetModel_1Dtemporal/results.json', '2D/1D', 'tab:green'),
    # ]
    # plot_comparison_side_by_side(
    #     left_data=None,
    #     right_data=None,
    #     left_xlabel=r'Block size $Q$',
    #     right_xlabel=r'Block size $Q$',
    #     left_ylabel='PSNR (dB)',
    #     right_ylabel='VRAM (GB)',
    #     save_path=f'{ma_base}/model_ablation.pdf',
    #     shared_legend=False,
    #     aspect_ratio=0.25,
    #     left_plot_fn=lambda ax: _plot_dict_series_on_axis(
    #         ax, ma_models, '0.05', 'psnr',
    #         r'Block size $Q$', 'PSNR (dB)',
    #     ),
    #     right_plot_fn=lambda ax: _plot_dict_series_on_axis(
    #         ax, ma_models, '0.05', 'memory',
    #         r'Block size $Q$', 'VRAM (GB)',
    #         transform=lambda v: v / 1024,
    #     ),
    # )


    # # --- Timestep sampling: side-by-side (accelerations + scheduling type) ---
    # ts_base = args.base / 'experiments/timestep_sampling'
    # plot_comparison_side_by_side(
    #     left_data=None,
    #     right_data=None,
    #     left_xlabel=r"Maximum Timestep $T'$",
    #     right_xlabel=r"Maximum Timestep $T'$",
    #     left_ylabel='PSNR (dB)',
    #     right_ylabel='PSNR (dB)',
    #     save_path=f'{ts_base}/timestep_ablation.pdf',
    #     shared_legend=False,
    #     aspect_ratio=0.45,
    #     left_plot_fn=lambda ax: _plot_multi_series_on_axis(
    #         ax, f'{ts_base}/type_simulation_setup/results.json',
    #         [('descending', 'descending', 'tab:green'),
    #          ('random', 'random', 'tab:orange')],
    #         r"Maximum Timestep $T'$", 'PSNR (dB)',
    #     ),
    #     right_plot_fn=lambda ax: _plot_multi_series_on_axis(
    #         ax, f'{ts_base}/accerlations/results.json',
    #         [('4', '4x acceleration', 'tab:blue'),
    #          ('8', '8x acceleration', 'tab:orange'),
    #          ('12', '12x acceleration', 'tab:green')],
    #         r"Maximum Timestep $T'$", 'PSNR (dB)',
    #         mark_max=True,
    #     ),
    # )


    # # --- Simulation setup (N=120, retrospective 12x): tradeoff scatter plots ---
    # # Table columns used: K, Q, PSNR [dB], VRAM [GB], Time [s]
    # sim_setup_points = [
    #     {'k': 100, 'q': 12, 'psnr': 31.48, 'vram': 12.3, 'time': 48.31},
    #     {'k': 100, 'q': 36, 'psnr': 33.80, 'vram': 21.0, 'time': 114.9},
    #     {'k': 100, 'q': 120, 'psnr': 34.26, 'vram': 39.5, 'time': 398.9},
    #     {'k': 200, 'q': 12, 'psnr': 33.49, 'vram': 12.3, 'time': 94.12},
    #     {'k': 200, 'q': 36, 'psnr': 34.26, 'vram': 20.3, 'time': 227.2},
    #     {'k': 200, 'q': 120, 'psnr': 34.59, 'vram': 38.8, 'time': 746.3},
    #     {'k': 400, 'q': 12, 'psnr': 33.94, 'vram': 12.3, 'time': 186.3},
    #     {'k': 400, 'q': 36, 'psnr': 34.22, 'vram': 20.8, 'time': 452.7},
    #     {'k': 400, 'q': 120, 'psnr': 34.59, 'vram': 39.2, 'time': 1459.0},
    # ]

    # plot_comparison_side_by_side(
    #     left_data=None,
    #     right_data=None,
    #     left_xlabel='Time (s)',
    #     right_xlabel='VRAM (GB)',
    #     left_ylabel='PSNR (dB)',
    #     right_ylabel='PSNR (dB)',
    #     save_path=args.base / 'experiments/tradeoff_time_vram_vs_psnr.pdf',
    #     shared_legend=False,
    #     aspect_ratio=0.28,
    #     left_plot_fn=lambda ax: _plot_tradeoff_scatter_on_axis(
    #         ax,
    #         sim_setup_points,
    #         x_key='time',
    #         xlabel='Reconstruction time (s)',
    #         ylabel='PSNR (dB)',
    #     ),
    #     right_plot_fn=lambda ax: _plot_tradeoff_scatter_on_axis(
    #         ax,
    #         sim_setup_points,
    #         x_key='vram',
    #         xlabel='VRAM (GB)',
    #         ylabel='PSNR (dB)',
    #     ),
    # )


    ### Statistical analysis of quantitative metrics

    # # # --- Multi-seed retrospective ---
    # seeds = [42, 1234, 1697, 1996, 2026]
    # base = args.base / 'experiments/multi_seed'
    # method_csv_groups = {
    #     "PDDR": [f"{base}/PDDR/retrospective/12/{s}/metrics.csv" for s in seeds],
    #     "L+S": [f"{base}/L+S/retrospective/12/{s}/metrics.csv" for s in seeds],
    #     "DPS": [f"{base}/DPS/retrospective/12/{s}/metrics.csv" for s in seeds],
    #     "SDR": [f"{base}/SDR/retrospective/12/{s}/metrics.csv" for s in seeds],
    #     "T-DIP": [f"{base}/TDIP/retrospective/12/{s}/metrics.csv" for s in seeds],
    #     # "FMLP": [f"{base}/FMLP/retrospective/12/{s}/metrics.csv" for s in seeds],
    #     "FMLP": [f"{base}/FMLP/retrospective/12/{s}/metrics.csv" for s in [42, 1996, 2026]],
    #     "ZF": [f"{base}/ZF/retrospective/12/{s}/metrics.csv" for s in seeds],
    #     "dSTDM": [f"{base}/dSTDM/192x192/retrospective/12/{s}/metrics.csv" for s in seeds],
    # }

    # compute_pairwise_method_statistics(
    #     method_csv_groups=method_csv_groups,
    #     output_csv_path=args.base / 'experiments/multi_seed/statistics/pddr_vs_ls_vs_dps_vs_sdr_vs_tdip_vs_fmlp_zf_dstdm_retrospective_12x.csv',
    # )


    # # --- Prospective short ---
    # compute_pairwise_method_statistics(
    #     metrics_csv_paths=[
    #     args.base / 'experiments/prospective_short/statistics/pddr.csv',
    #     args.base / 'experiments/prospective_short/statistics/lps.csv',
    #     args.base / 'experiments/prospective_short/statistics/tdip.csv',
    #     args.base / 'experiments/prospective_short/statistics/fmlp.csv',
    #     args.base / 'experiments/prospective_short/statistics/dps.csv',
    #     args.base / 'experiments/prospective_short/statistics/sdr.csv',
    #     args.base / 'experiments/prospective_short/statistics/dstdm.csv',
    #     args.base / 'experiments/prospective_short/statistics/zf.csv'
    #     ],
    #     method_names=["PDDR", "L+S", "T-DIP", "FMLP", "DPS", "SDR", "dSTDM", "ZF"],
    #     output_csv_path=args.base / 'experiments/prospective_short/statistics/pddr_vs_lps_vs_tdip_vs_fmlp_vs_dps_vs_sdr_vs_dstdm_vs_zf_prospective.csv',
    # )

    # # --- ACS 12 ---
    # compute_pairwise_method_statistics(
    #     metrics_csv_paths=[
    #     args.base / 'experiments/ACS_12/PDDR/retrospective/100_steps/12/metrics.csv',
    #     args.base / 'experiments/ACS_12/L+S/retrospective/12/metrics.csv',
    #     args.base / 'experiments/ACS_12/DPS_nograd/retrospective/12/metrics.csv',
    #     args.base / 'experiments/ACS_12/SDR/retrospective/12/metrics.csv',
    #     args.base / 'experiments/ACS_12/TDIP/retrospective/12/metrics.csv',
    #     args.base / 'experiments/ACS_12/FMLP/retrospective/12/metrics.csv', 
    #     args.base / 'experiments/ACS_12/ZF/retrospective/12/metrics.csv'
    #     ],
    #     method_names=["PDDR", "L+S", "DPS", "SDR", "T-DIP", "FMLP", "ZF"],
    #     output_csv_path=args.base / 'experiments/ACS_12/statistics/ACS_12_retrospective_12x.csv',
    # )


if __name__ == "__main__":
    main()
