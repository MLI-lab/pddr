import argparse
import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr

from torch.utils import data
from pathlib import Path

from cardiac_diffusion.mri import fftnc, ifftnc, kspace2image, image2kspace
from cardiac_diffusion.utils import tensor_to_gif


class CMRxCycle(data.Dataset):
    def __init__(
            self, 
            data_path, 
            repeat_cycle=0,
        ):
        super().__init__()
        self.data_path = data_path
        self.repeat_cycle = repeat_cycle

        # check all files
        if isinstance(data_path, (list, tuple)):
            paths = []
            for p in data_path:
                p = Path(p)
                if p.is_file():
                    paths.append(p)
                else:
                    raise ValueError(f"{p} is not a valid file.")
        else:
            raise ValueError(f"{data_path} is not a valid testset.")


        # split each file to return slices individually
        self.raw_samples = []
        for fname in paths:
            with h5py.File(fname, "r") as hf:
                num_slices = hf['kspace'].shape[1]  # count slices in [t, s, ky, kx] or [t, s, coil, ky, kx]  

            if 'sax' in fname.stem.lower():
                middle_slice = True
            else:
                middle_slice = False

            new_raw_samples = []
            if middle_slice:
                raw_sample = (fname, num_slices // 2)
                new_raw_samples.append(raw_sample)
            else:
                for slice_ind in range(num_slices):
                    raw_sample = (fname, slice_ind)
                    new_raw_samples.append(raw_sample)
            self.raw_samples += new_raw_samples

    def _normalize(self, x, kspace_norm):
        x = x * np.sqrt(2*x.shape[0]*np.prod(x.shape[-2:])) / kspace_norm
        return x

    def __len__(self):
        return len(self.raw_samples)

    def __getitem__(self, idx):
        fname, dataslice = self.raw_samples[idx]

        with h5py.File(fname, "r") as hf:
            # kspace = hf['kspace'][:, dataslice]                                      # [frame, coil, ky, kx] or [frame, ky, kx]
            ground = hf['image'][:, dataslice]                                       # [frame, ky, kx] -- complex valued
            sensitivities = hf['sensitivities'][dataslice]                           # [coil, ky, kx]
            
        # kspace = torch.from_numpy(kspace)
        ground = torch.from_numpy(ground)
        sensitivities = torch.from_numpy(sensitivities)

        name = f'{fname.parent.stem}_{dataslice}'
        return ground, sensitivities, name
    

def alternate_cycle(cycle, factor=1.0, phase_aware=True):
    """
    Alternate the duration of the cycle by speeding up or slowing down the cycle. 
    This is done by resampling the cycle using linear interpolation between the original frames, insertion and deletion of frames. 

    Args:
        cycle: A complex-valued tensor of shape [T, ...] representing the cycle.
        factor: A float representing the factor by which to speed up (>1) or slow down (<1) the cycle.
        phase_aware: If True and cycle is complex-valued, interpolate magnitude and phase
            separately with shortest-path phase interpolation.
    """
    if cycle.ndim < 1:
        raise ValueError("cycle must have at least one dimension")
    if factor <= 0:
        raise ValueError("factor must be positive")

    num_frames = cycle.shape[0]
    if num_frames == 1:
        return cycle.clone()

    target_frames = int(np.round((num_frames - 1) / factor)) + 1
    target_frames = max(target_frames, 2)

    if target_frames == num_frames and np.isclose(factor, 1.0):
        return cycle.clone()

    timeline = torch.linspace(
        0,
        num_frames - 1,
        steps=target_frames,
        device=cycle.device,
        dtype=torch.float32,
    )

    left_idx = torch.floor(timeline).long()
    right_idx = torch.clamp(left_idx + 1, max=num_frames - 1)
    alpha = (timeline - left_idx.to(timeline.dtype)).view(-1, *([1] * (cycle.ndim - 1)))


    left = cycle[left_idx]
    right = cycle[right_idx]

    if phase_aware and torch.is_complex(cycle):
        alpha_real = alpha.to(left.real.dtype)
        mag = torch.lerp(torch.abs(left), torch.abs(right), alpha_real)

        left_phase = torch.angle(left)
        right_phase = torch.angle(right)
        phase_delta = torch.atan2(torch.sin(right_phase - left_phase), torch.cos(right_phase - left_phase))
        phase = left_phase + alpha_real * phase_delta
        return torch.polar(mag, phase)

    return left + (right - left) * alpha.to(left.dtype)


def add_motion(video, respiratory_cycles=1.0, x_factor=0.02, y_factor=0.01):
    """
    Add synthetic respiratory motion to the video by applying a sinusoidal shift to the frames. 

    Args:
        video: A complex-valued tensor of shape [T, H, W] representing the video frames.
        respiratory_cycles: A float representing the number of respiratory cycles over the entire video duration.
        x_factor: A float representing the maximum x-shift as a fraction of the image size.
        y_factor: A float representing the maximum y-shift as a fraction of the image size.
    """
    if video.ndim != 3:
        raise ValueError(f"video must have shape [T, H, W], got {tuple(video.shape)}")

    num_frames, height, width = video.shape
    t = torch.arange(num_frames, device=video.device, dtype=torch.float32) / num_frames
    shift_y = y_factor * height * torch.sin(2 * np.pi * respiratory_cycles * t)
    shift_x = x_factor * width * torch.cos(2 * np.pi * respiratory_cycles * t)

    moved_frames = []
    for frame_idx in range(num_frames):
        shift_y_int = int(torch.round(shift_y[frame_idx]).item())
        shift_x_int = int(torch.round(shift_x[frame_idx]).item())

        moved_frame = torch.roll(video[frame_idx], shifts=(shift_y_int, shift_x_int), dims=(-2, -1))
        moved_frames.append(moved_frame)

    return torch.stack(moved_frames, dim=0)


def _create_pause_segment(cycle, num_frames, phase_aware=True):
    if num_frames <= 0:
        return cycle[:0]
    if cycle.shape[0] == 1:
        return cycle[-1:].repeat(num_frames, *([1] * (cycle.ndim - 1)))

    first_frame = cycle[0]
    second_last_frame = cycle[-2]
    last_frame = cycle[-1]

    def _midpoint(a, b):
        alpha = 0.5
        if not torch.is_complex(cycle):
            return torch.lerp(a, b, alpha)

        if phase_aware:
            mag = torch.lerp(a.abs(), b.abs(), alpha)
            a_phase = torch.angle(a)
            b_phase = torch.angle(b)
            phase_delta = torch.atan2(torch.sin(b_phase - a_phase), torch.cos(b_phase - a_phase))
            phase = a_phase + alpha * phase_delta
            return torch.polar(mag, phase)

        return torch.lerp(a, b, alpha)

    anchor_frames = torch.stack([
        last_frame,
        _midpoint(last_frame, second_last_frame),
        second_last_frame,
        _midpoint(second_last_frame, first_frame),
        first_frame,
    ], dim=0)

    if num_frames == anchor_frames.shape[0]:
        return anchor_frames

    if num_frames < anchor_frames.shape[0]:
        return anchor_frames[:num_frames]

    repeats = int(np.ceil(num_frames / anchor_frames.shape[0]))
    expanded = anchor_frames.repeat((repeats,) + (1,) * (cycle.ndim - 1))
    return expanded[:num_frames]


def create_non_periodic_video(
    cycle,
    num_frames,
    factor_std=0.15,
    phase_aware=True,
    apply_motion=False,
    respiratory_cycles=1.0,
    respiratory_cycles_std=0.1,
    motion_x_factor=0.01,
    motion_y_factor=0.01,
    apply_arrhythmia=False,
    arrhythmia_factor_mean=4.0,
    arrhythmia_factor_std=0.5,
    arrhythmia_pause_frames=6,
):
    """
    Create a non-periodic video by alternating and repeating the cycle. 
    Each cycle duration is alternated by a random factor sampled from a normal distribution with mean 1.0 and std factor_std.
    Optionally, synthetic respiratory motion can be applied to the full video, using a number of respiratory cycles defined by given mean and std.
    Optionally, short abnormal cycles can be inserted into the video to simulate arrhythmia.

    Args:
        cycle: A complex-valued tensor of shape [T, ...] representing the cycle.
        num_frames: An integer representing the total number of frames in the output video.
        factor_std: A float representing the std of the factor by which to speed up (>1) or slow down (<1) the individual cycles.
        phase_aware: If True and cycle is complex-valued, use phase-aware interpolation.
        apply_motion: If True, apply synthetic respiratory motion after the non-periodic video is assembled.
        respiratory_cycles: Mean number of respiratory cycles across the full video duration.
        respiratory_cycles_std: Std of the respiratory cycle count used when sampling motion variation.
        motion_x_factor: Max x-shift as fraction of image width for motion.
        motion_y_factor: Max y-shift as fraction of image height for motion.
        apply_arrhythmia: If True, periodically insert a short abnormal cycle.
        arrhythmia_factor_mean: Mean factor used for the abnormal short cycle.
        arrhythmia_factor_std: Std factor used for the abnormal short cycle.
        arrhythmia_pause_frames: Number of frames to hold a short post-arrhythmia pause.
    """
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")

    segments = []
    total_frames = 0
    normal_segments_since_abnormal = 0
    next_abnormal_after = int(np.random.choice([2, 3, 4])) if apply_arrhythmia else None

    while total_frames < num_frames:
        factor = float(np.random.normal(loc=1.0, scale=factor_std))
        if factor <= 0:
            continue

        varied_cycle = alternate_cycle(cycle, factor=factor, phase_aware=phase_aware)
        segments.append(varied_cycle)
        total_frames += varied_cycle.shape[0]
        normal_segments_since_abnormal += 1

        if apply_arrhythmia and total_frames < num_frames and normal_segments_since_abnormal >= next_abnormal_after:
            abnormal_factor = float(np.random.normal(loc=arrhythmia_factor_mean, scale=arrhythmia_factor_std))
            while abnormal_factor <= 0:
                abnormal_factor = float(np.random.normal(loc=arrhythmia_factor_mean, scale=arrhythmia_factor_std))

            abnormal_cycle = alternate_cycle(cycle, factor=abnormal_factor, phase_aware=phase_aware)
            segments.append(abnormal_cycle)
            total_frames += abnormal_cycle.shape[0]

            if arrhythmia_pause_frames > 0 and total_frames < num_frames:
                pause_segment = _create_pause_segment(
                    cycle,
                    num_frames=arrhythmia_pause_frames,
                    phase_aware=phase_aware,
                )
                segments.append(pause_segment)
                total_frames += pause_segment.shape[0]

            normal_segments_since_abnormal = 0
            next_abnormal_after = int(np.random.choice([2, 3, 4]))

    video = torch.cat(segments, dim=0)
    video = video[:num_frames]

    if apply_motion:
        sampled_respiratory_cycles = float(np.random.normal(loc=respiratory_cycles, scale=respiratory_cycles_std))
        while sampled_respiratory_cycles <= 0:
            sampled_respiratory_cycles = float(np.random.normal(loc=respiratory_cycles, scale=respiratory_cycles_std))

        video = add_motion(
            video,
            respiratory_cycles=sampled_respiratory_cycles,
            x_factor=motion_x_factor,
            y_factor=motion_y_factor,
        )

    return video


def create_mri_sequence(video, sensitivities, save_h5=False, save_path=None):
    """
    Create a multi-coil MRI sequence by applying the coil sensitivities to the video frames and transforming them to k-space. 

    Args:
        video: A complex-valued tensor of shape [T, H, W] representing the video frames.
        sensitivities: A complex-valued tensor of shape [C, H, W] representing the coil sensitivities.
    """
    if video.ndim != 3:
        raise ValueError(f"video must have shape [T, H, W], got {tuple(video.shape)}")
    if sensitivities.ndim != 3:
        raise ValueError(f"sensitivities must have shape [C, H, W], got {tuple(sensitivities.shape)}")

    if not torch.is_complex(video):
        video = video.to(torch.complex64)
    if not torch.is_complex(sensitivities):
        sensitivities = sensitivities.to(torch.complex64)

    kspace = image2kspace(video.unsqueeze(1), sensitivities.unsqueeze(0), dim=(2, 3))
    video_out = video.unsqueeze(1)
    kspace_out = kspace.unsqueeze(1)
    sensitivities_out = sensitivities.unsqueeze(0)

    if save_h5 is not False:
        if save_path is None and isinstance(save_h5, (str, Path)):
            save_path = save_h5
        if save_path is None:
            raise ValueError("save_path must be provided when save_h5 is enabled")

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(save_path, "w") as hf:
            hf.create_dataset("kspace", data=kspace_out.cpu().numpy())
            hf.create_dataset("sensitivities", data=sensitivities_out.cpu().numpy())
            hf.create_dataset("image", data=video_out.cpu().numpy())

    return video_out, kspace_out, sensitivities_out

## plots
def plot_cycle_alternation(original_cycle, alternated_cycle, factor=None, gif_path=None):
    original_mag = original_cycle.abs().detach().cpu()
    alternated_mag = alternated_cycle.abs().detach().cpu()

    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    original_idx = [0, original_mag.shape[0] - 1]
    alternated_idx = [0, alternated_mag.shape[0] - 1]

    for ax, frame_idx in zip(axes[0], original_idx):
        ax.imshow(original_mag[frame_idx], cmap="gray")
        ax.set_title(f"Original frame {frame_idx}")
        ax.axis("off")

    for ax, frame_idx in zip(axes[1], alternated_idx):
        ax.imshow(alternated_mag[frame_idx], cmap="gray")
        ax.set_title(f"Alternated frame {frame_idx}")
        ax.axis("off")

    title = "Original vs. alternated cycle"
    if factor is not None:
        title += f" (factor={factor:.3f})"
    fig.suptitle(title)
    fig.tight_layout()
    plt.show()
    plt.close(fig)

    if gif_path is not None:
        gif_path = Path(gif_path)
        gif_path.parent.mkdir(parents=True, exist_ok=True)
        max_frames = max(original_mag.shape[0], alternated_mag.shape[0])

        if original_mag.shape[0] < max_frames:
            pad = torch.zeros(
                (max_frames - original_mag.shape[0],) + original_mag.shape[1:],
                dtype=original_mag.dtype,
            )
            original_mag = torch.cat([original_mag, pad], dim=0)

        if alternated_mag.shape[0] < max_frames:
            pad = torch.zeros(
                (max_frames - alternated_mag.shape[0],) + alternated_mag.shape[1:],
                dtype=alternated_mag.dtype,
            )
            alternated_mag = torch.cat([alternated_mag, pad], dim=0)

        preview = torch.cat([
            original_mag.unsqueeze(1),
            alternated_mag.unsqueeze(1),
        ], dim=2)
        tensor_to_gif(preview, gif_path, duration=200)


def _repeat_cycle_to_length(cycle, target_frames):
    if cycle.shape[0] >= target_frames:
        return cycle[:target_frames]

    repeats = int(np.ceil(target_frames / cycle.shape[0]))
    repeated = cycle.repeat((repeats,) + (1,) * (cycle.ndim - 1))
    return repeated[:target_frames]


def plot_full_video_preview(original_cycle, simulated_video, gif_path=None):
    original_mag = original_cycle.abs().detach().cpu()
    simulated_mag = simulated_video.abs().detach().cpu()
    original_mag = _repeat_cycle_to_length(original_mag, simulated_mag.shape[0])

    if gif_path is None:
        return

    gif_path = Path(gif_path)
    gif_path.parent.mkdir(parents=True, exist_ok=True)

    preview = torch.cat([
        original_mag.unsqueeze(1),
        simulated_mag.unsqueeze(1),
    ], dim=2)
    tensor_to_gif(preview, gif_path, duration=200)


def inspect_data(kspace, sensitivities, video):
    """
    print infos and plot data (kspace, sensitivities,  video) for sanity check
    """
    ####
    print('kspace:')
    print(f'shape: {kspace.shape}, [frame, slice, coil, ky, kx]')
    print(f'dtype: {kspace.dtype}, complex64')
    # fidx = kspace.shape[0] // 2
    # sidx = kspace.shape[1] // 2
    fidx = 0
    sidx = torch.randint(0, kspace.shape[1], (1,)).item()

    num_coils = kspace.shape[2]
    num_plots = min(num_coils, 10)
    rows = int(np.ceil(num_plots / 2))
    fig, axes = plt.subplots(rows, 2, squeeze=False)
    for i, ax in enumerate(axes.ravel()):
        if i >= num_plots:
            ax.axis('off')
            continue
        ax.imshow(abs(kspace[fidx, sidx, i]), cmap='gray', norm=clr.PowerNorm(gamma=0.25))
        ax.axis('off')
    plt.show()
    plt.close(fig)
    ####
    print('sensitivity maps:')
    print(f'shape: {sensitivities.shape}, [slice, coil, ky, kx]')
    print(f'dtype: {sensitivities.dtype}, complex64')

    num_coils = sensitivities.shape[1]
    num_plots = min(num_coils, 10)
    rows = int(np.ceil(num_plots / 2))
    fig, axes = plt.subplots(rows, 2, squeeze=False)
    for i, ax in enumerate(axes.ravel()):
        if i >= num_plots:
            ax.axis('off')
            continue
        ax.imshow(abs(sensitivities[sidx, i]), cmap='gray')
        ax.axis('off')
    plt.show()
    plt.close(fig)
    ####
    print('MVUE images:')
    print(f'shape: {video.shape}, [frame, slice, ky, kx]')
    print(f'dtype: {video.dtype}, complex64')

    num_frames = video.shape[0]
    num_cols = min(6, num_frames)
    num_rows = int(np.ceil(num_frames / num_cols))
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(2.2 * num_cols, 2.2 * num_rows), squeeze=False)
    for frame_idx, ax in enumerate(axes.ravel()):
        if frame_idx >= num_frames:
            ax.axis('off')
            continue
        ax.imshow(abs(video[frame_idx, sidx]), cmap='gray')
        ax.set_title(f'F{frame_idx}')
        ax.axis('off')
    plt.tight_layout()
    plt.show()
    plt.close(fig)

    plt.imshow(abs(video[fidx, sidx]), cmap='gray')
    plt.axis('off')
    plt.show()

    print('kspace -> image invertibility check (adaptive combine):')
    if torch.is_tensor(kspace):
        kspace_tensor = kspace
    else:
        kspace_tensor = torch.from_numpy(kspace)

    if torch.is_tensor(sensitivities):
        sens_tensor = sensitivities
    else:
        sens_tensor = torch.from_numpy(sensitivities)

    if torch.is_tensor(video):
        video_tensor = video
    else:
        video_tensor = torch.from_numpy(video)

    recon_video = kspace2image(
        kspace_tensor,
        fdim=(3, 4),
        cdim=2,
        adaptive=True,
        sens_maps=sens_tensor,
    )

    # sens_energy = torch.sum(torch.abs(sens_tensor) ** 2, dim=1)
    # recon_video_corrected = recon_video / torch.clamp(sens_energy.unsqueeze(0), min=1e-8)
    recon_video_corrected = recon_video

    max_abs_err = torch.max(torch.abs(recon_video_corrected - video_tensor)).item()
    print(f'max abs error: {max_abs_err:.3e}')
    assert torch.allclose(recon_video_corrected, video_tensor, rtol=1e-4, atol=1e-5), (
        f'kspace2image mismatch: max abs error={max_abs_err:.3e}'
    )

    num_frames = video_tensor.shape[0]
    compare_frames = list(range(num_frames))
    num_cols = 3
    num_rows = len(compare_frames)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(9, 2.4 * num_rows), squeeze=False)
    for row, frame_idx in enumerate(compare_frames):
        axes[row, 0].imshow(abs(video_tensor[frame_idx, sidx]), cmap='gray')
        axes[row, 0].set_title(f'Orig F{frame_idx}')
        axes[row, 0].axis('off')

        axes[row, 1].imshow(abs(recon_video_corrected[frame_idx, sidx]), cmap='gray')
        axes[row, 1].set_title(f'Recon F{frame_idx}')
        axes[row, 1].axis('off')

        diff = torch.abs(video_tensor[frame_idx, sidx] - recon_video_corrected[frame_idx, sidx])
        axes[row, 2].imshow(diff, cmap='magma')
        axes[row, 2].set_title(f'|Diff| F{frame_idx}')
        axes[row, 2].axis('off')
    plt.tight_layout()
    plt.show()
    plt.close(fig)

    print('RSS image as reference:')
    rssi = kspace2image(kspace_tensor, fdim=(3, 4), cdim=2, adaptive=False).cpu().numpy()
    plt.imshow(abs(rssi[fidx, sidx]), cmap='gray')
    plt.axis('off')
    plt.show()
    ####


def main():
    parser = argparse.ArgumentParser(description="Simulate non-periodic cardiac MRI cycles.", allow_abbrev=False)
    parser.add_argument(
        "--data-path",
        nargs="+",
        default=[# mixed_testset_mid
                # SAX (30)
                'datasets/CineProcessed/2023/TestSet/P002/cine_sax.h5',
                'datasets/CineProcessed/2023/TestSet/P006/cine_sax.h5',
                'datasets/CineProcessed/2023/TestSet/P013/cine_sax.h5',
                'datasets/CineProcessed/2023/TestSet/P020/cine_sax.h5',
                'datasets/CineProcessed/2023/TestSet/P022/cine_sax.h5',
                'datasets/CineProcessed/2023/TestSet/P025/cine_sax.h5',
                'datasets/CineProcessed/2023/TestSet/P028/cine_sax.h5',
                'datasets/CineProcessed/2023/TestSet/P032/cine_sax.h5',
                'datasets/CineProcessed/2023/TestSet/P036/cine_sax.h5',
                'datasets/CineProcessed/2023/TestSet/P041/cine_sax.h5',
                'datasets/CineProcessed/2023/TestSet/P046/cine_sax.h5',
                'datasets/CineProcessed/2023/TestSet/P047/cine_sax.h5',
                'datasets/CineProcessed/2023/TestSet/P049/cine_sax.h5',
                'datasets/CineProcessed/2023/TestSet/P055/cine_sax.h5',
                'datasets/CineProcessed/2023/TestSet/P056/cine_sax.h5',
                'datasets/CineProcessed/2023/TestSet/P060/cine_sax.h5',
                'datasets/CineProcessed/2023/TestSet/P066/cine_sax.h5',
                'datasets/CineProcessed/2023/TestSet/P067/cine_sax.h5',
                'datasets/CineProcessed/2023/TestSet/P072/cine_sax.h5',
                'datasets/CineProcessed/2023/TestSet/P075/cine_sax.h5',
                'datasets/CineProcessed/2023/TestSet/P077/cine_sax.h5',
                'datasets/CineProcessed/2023/TestSet/P079/cine_sax.h5',
                'datasets/CineProcessed/2023/TestSet/P082/cine_sax.h5',
                'datasets/CineProcessed/2023/TestSet/P087/cine_sax.h5',
                'datasets/CineProcessed/2023/TestSet/P088/cine_sax.h5',
                'datasets/CineProcessed/2023/TestSet/P089/cine_sax.h5',
                'datasets/CineProcessed/2023/TestSet/P094/cine_sax.h5',
                'datasets/CineProcessed/2023/TestSet/P097/cine_sax.h5',
                'datasets/CineProcessed/2023/TestSet/P107/cine_sax.h5',
                'datasets/CineProcessed/2023/TestSet/P110/cine_sax.h5',
                # LAX (5 a 3 views = 15)
                'datasets/CineProcessed/2023/TestSet/P005/cine_lax.h5',
                'datasets/CineProcessed/2023/TestSet/P020/cine_lax.h5',
                'datasets/CineProcessed/2023/TestSet/P022/cine_lax.h5',
                'datasets/CineProcessed/2023/TestSet/P036/cine_lax.h5',
                'datasets/CineProcessed/2023/TestSet/P040/cine_lax.h5',
                ],
        help="One or more HDF5 files for CMRxCycle.",
    )
    parser.add_argument("--h5-path", type=str, required=True, help="Output HDF5 path for the synthesized sequence.")
    parser.add_argument("--num-frames", type=int, default=60, help="Total number of frames to synthesize.")
    parser.add_argument("--factor-std", type=float, default=0.15, help="Std of the Gaussian used for multi-cycle variation.")
    parser.add_argument("--gif-path", type=str, default=None, help="Optional path for a GIF preview.")
    parser.add_argument("--no-display", type=bool, default=True, help="Do not open matplotlib windows.")
    parser.add_argument("--phase-aware", type=bool, default=True, help="Use phase-aware interpolation for complex-valued cycles.")
    parser.add_argument("--add-motion", action="store_true", help="Apply synthetic respiratory motion to the simulated video.")
    parser.add_argument("--respiratory-cycles", type=float, default=1.0, help="Number of respiratory cycles over the full video.")
    parser.add_argument("--respiratory-cycles-std", type=float, default=0.1, help="Std of the respiratory cycles used when motion is enabled.")
    parser.add_argument("--motion-x-factor", type=float, default=0.01, help="Max x-shift as fraction of image width.")
    parser.add_argument("--motion-y-factor", type=float, default=0.005, help="Max y-shift as fraction of image height.")
    parser.add_argument("--arrhythmia", action="store_true", help="Insert periodic abnormal short cycles into the video.")
    parser.add_argument("--arrhythmia-factor-mean", type=float, default=3.0, help="Mean factor used for abnormal short cycles.")
    parser.add_argument("--arrhythmia-factor-std", type=float, default=0.5, help="Std of the factor used for abnormal short cycles.")
    parser.add_argument("--arrhythmia-pause-frames", type=int, default=5, help="Number of frames to hold after each abnormal beat.")
    args, _ = parser.parse_known_args()

    dataset = CMRxCycle(args.data_path)
    dataloader = data.DataLoader(dataset, batch_size=1, shuffle=False)

    output_dir = None
    if args.h5_path is not None:
        output_dir = Path(args.h5_path)
        output_dir.mkdir(parents=True, exist_ok=True)

    for sample_idx, batch in enumerate(dataloader):
        ground, sensitivities, name = batch
        ground = ground.squeeze(0)
        sensitivities = sensitivities.squeeze(0)
        name = name[0]

        ground = ground.to(torch.complex64) if not torch.is_complex(ground) else ground
        sensitivities = sensitivities.to(torch.complex64) if not torch.is_complex(sensitivities) else sensitivities

        video = create_non_periodic_video(
            ground,
            num_frames=args.num_frames,
            factor_std=args.factor_std,
            phase_aware=args.phase_aware,
            apply_motion=args.add_motion,
            respiratory_cycles=args.respiratory_cycles,
            respiratory_cycles_std=args.respiratory_cycles_std,
            motion_x_factor=args.motion_x_factor,
            motion_y_factor=args.motion_y_factor,
            apply_arrhythmia=args.arrhythmia,
            arrhythmia_factor_mean=args.arrhythmia_factor_mean,
            arrhythmia_factor_std=args.arrhythmia_factor_std,
            arrhythmia_pause_frames=args.arrhythmia_pause_frames,
        )

        if args.gif_path is not None:
            full_gif_path = Path(args.gif_path) / f"{name}.gif"
            full_gif_path = full_gif_path.with_name(f"{full_gif_path.stem}_full{full_gif_path.suffix}")
            plot_full_video_preview(ground, video, gif_path=full_gif_path)

        save_path = None
        if output_dir is not None:
            save_path = str(output_dir / f"{name}.h5")

        video, kspace, sensitivities = create_mri_sequence(
            video,
            sensitivities,
            save_h5=output_dir is not None,
            save_path=save_path,
        )

        if not args.no_display:
            inspect_data(kspace, sensitivities, video)

        if output_dir is not None:
            print(f"Saved synthesized sequence for {name} to {save_path}")


if __name__ == "__main__":
    main()
