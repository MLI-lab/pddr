import numpy as np
import torch

from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
from skimage.metrics import structural_similarity, peak_signal_noise_ratio

from cardiac_diffusion.mri import kspace2image, image2kspace


def compute_metrics(ground_truth, reconstruction, return_image=True):
    """
    Compute SSIM, PSNR, and NMSE for a batch of real-valued images.
    Args:
        ground_truth (torch.Tensor): Ground truth images in shape (B, C, H, W).
        reconstruction (torch.Tensor): Reconstructed images in shape (B, C, H, W).
        return_image (bool): Whether to return images with metrics overlaid.
    Returns:
        list: SSIM values for each image in the batch.
        list: PSNR values for each image in the batch.
        list: NMSE values for each image in the batch.
        torch.Tensor (optional): Images with metrics overlaid in shape (B, C, H, W).
    """
    b, c, h, w = ground_truth.shape
    b_ssims, b_psnrs, b_nmses = [], [], []

    if return_image:
        b_images = []

    for i in range(b):
        # Convert tensor to numpy array for metric computation
        ground_truth_np = ground_truth[i].cpu().numpy()
        reconstruction_np = reconstruction[i].cpu().numpy()

        # Normalize
        ground_truth_np = (ground_truth_np - ground_truth_np.min()) / (ground_truth_np.max() - ground_truth_np.min())
        reconstruction_np = (reconstruction_np - reconstruction_np.min()) / (reconstruction_np.max() - reconstruction_np.min())

        # Compute metrics
        ssim = structural_similarity(ground_truth_np, reconstruction_np, data_range=1.0, channel_axis=0)
        psnr = peak_signal_noise_ratio(ground_truth_np, reconstruction_np, data_range=1.0)
        nmse = np.array(np.linalg.norm(ground_truth_np - reconstruction_np) ** 2 / np.linalg.norm(ground_truth_np) ** 2)
        b_ssims.append(ssim)
        b_psnrs.append(psnr)
        b_nmses.append(nmse)

        if return_image:
            # Convert numpy to PIL Image
            to_pil = transforms.ToPILImage()
            reconstruction_pil = to_pil(reconstruction_np.transpose(1, 2, 0))

            # Draw metrics on the image
            font = ImageFont.load_default(size=18)
            draw = ImageDraw.Draw(reconstruction_pil)
            draw.text((0, 0), f"SSIM={format(ssim*100, '.1f')}%", fill='yellow', font=font)
            draw.text((0, 18), f"PSNR={format(psnr, '.1f')}dB", fill='yellow', font=font)
            to_tensor = transforms.ToTensor()
            metric_image = to_tensor(reconstruction_pil)
            b_images.append(metric_image)

    if return_image:
        metric_images = torch.stack(b_images, dim=0)
        return b_ssims, b_psnrs, b_nmses, metric_images

    return b_ssims, b_psnrs, b_nmses


def ser(signal, estimate, normalized=False):
    """
    Compute the Signal-to-Error Ratio (SER) between two tensors.
    Args:
        signal (torch.Tensor): The signal tensor.
        estimate (torch.Tensor): The estimated tensor of the same shape as signal.
    Returns:
        float: The SER value in decibels (dB).
    """
    assert signal.shape == estimate.shape, 'Signal and estimate must have the same shape.'

    # Compute SER
    power_signal = torch.sum(torch.square(signal))
    if normalized:
        power_estimate = torch.sum(torch.square(estimate))
        signal = signal / power_signal
        estimate = estimate / power_estimate
        power_signal = torch.sum(torch.square(signal))
    power_error = torch.sum(torch.square(estimate - signal))
    
    if power_error == 0:
        return float('inf')  # Infinite SER if there is no error

    ser_value = 10 * torch.log10(power_signal / power_error)
    return ser_value.item()


def compute_ser(reconstruction, mask, measurements, sensitivities):
    assert reconstruction.ndim == 3, f"Expected reconstruction to have 3 dimensions (complex-valued: f, h, w), got {reconstruction.ndim}"

    masked_kspace = image2kspace(reconstruction.unsqueeze(1), sensitivities.unsqueeze(0), dim=(2, 3)) * mask
    masked_kspace = torch.stack((masked_kspace.real, masked_kspace.imag), dim=0)
    measurements = torch.stack((measurements.real, measurements.imag), dim=0)

    return ser(measurements, masked_kspace)


def _make_torch_generator(seed, device):
    if seed is None:
        return None

    if device.type == 'cuda':
        generator = torch.Generator(device='cuda')
    else:
        generator = torch.Generator()

    generator.manual_seed(int(seed))
    return generator


def split_mask_and_kspace(mask, kspace, validation_lines=2, sparse_representation=False, seed=None):
    """
    Helper for SER evaluation.
    Splits the mask into training and validation masks, and applies the masks to the k-space data.
    """
    
    generator = _make_torch_generator(seed, mask.device)

    if sparse_representation:
        assert mask.ndim == 5, f"Expected mask to have 5 dimensions, got {mask.ndim}"
        assert kspace.ndim == 6, f"Expected kspace to have 6 dimensions, got {kspace.ndim}"
        assert mask.shape[2] == kspace.shape[4], f"Line dimension mismatch between mask and kspace"

        total_lines = mask.shape[2]
        training_count = total_lines - validation_lines

        # Choose validation indices at random
        perm = torch.randperm(total_lines, device=mask.device, generator=generator)
        validation_indices = torch.sort(perm[:validation_lines])[0]
        training_indices = torch.sort(perm[validation_lines:])[0]

        # Use gather along dimension 2 for mask
        training_idx_exp = training_indices.view(1, 1, -1, 1, 1).expand(mask.size(0), mask.size(1), training_count, mask.size(3), mask.size(4))
        validation_idx_exp = validation_indices.view(1, 1, -1, 1, 1).expand(mask.size(0), mask.size(1), validation_lines, mask.size(3), mask.size(4))
        training_mask = torch.gather(mask, 2, training_idx_exp)
        validation_mask = torch.gather(mask, 2, validation_idx_exp)

        # For kspace, the corresponding line dimension is at index 4
        training_idx_exp_ks = training_indices.view(1, 1, 1, 1, -1).unsqueeze(5).expand(kspace.size(0), kspace.size(1), kspace.size(2), kspace.size(3), training_count, kspace.size(5))
        validation_idx_exp_ks = validation_indices.view(1, 1, 1, 1, -1).unsqueeze(5).expand(kspace.size(0), kspace.size(1), kspace.size(2), kspace.size(3), validation_lines, kspace.size(5))
        training_kspace = torch.gather(kspace, 4, training_idx_exp_ks)
        validation_kspace = torch.gather(kspace, 4, validation_idx_exp_ks)

        # # testing:
        # print("Training mask shape:", training_mask.shape)
        # for i in range(training_count):
        #     assert torch.all(training_mask[:, :, i, :, :] == mask[:, :, training_indices[i], :, :]), f"Mismatch in training mask at index {i}"
        # print("Validation mask shape:", validation_mask.shape)
        # for i in range(validation_lines):
        #     assert torch.all(validation_mask[:, :, i, :, :] == mask[:, :, validation_indices[i], :, :]), f"Mismatch in validation mask at index {i}"
        # print("Training kspace shape:", training_kspace.shape)
        # for i in range(training_count):
        #     assert torch.all(training_kspace[:, :, :, :, i, :] == kspace[:, :, :, :, training_indices[i], :]), f"Mismatch in training kspace at index {i}"
        # print("Validation kspace shape:", validation_kspace.shape)
        # for i in range(validation_lines):
        #     assert torch.all(validation_kspace[:, :, :, :, i, :] == kspace[:, :, :, :, validation_indices[i], :]), f"Mismatch in validation kspace at index {i}"

        return training_mask, training_kspace, validation_mask, validation_kspace
    else:
        assert mask.ndim == 5, f"Expected mask to have 5 dimensions, got {mask.ndim}"
        B, F, _, H, W = mask.shape
        training_mask = mask.clone()
        validation_mask = torch.zeros_like(mask)
        # loops because one could have acquired different lines in different frames
        for b in range(B):
            for f in range(F):
                acquired_rows = (mask[b, f, 0] == 1).any(dim=1).nonzero(as_tuple=False).squeeze(1)
                if acquired_rows.numel() < validation_lines:
                    if f == F-1: break
                    raise ValueError("Not enough acquired rows for validation")
                perm = torch.randperm(acquired_rows.numel(), device=mask.device, generator=generator)
                val_rows = acquired_rows[perm[:validation_lines]]
                # Zero out the selected rows in training mask
                training_mask[b, f, :, val_rows, :] = 0
                # Validation mask keeps only the selected rows
                validation_mask[b, f, :, val_rows, :] = mask[b, f, :, val_rows, :]

        if kspace.ndim == 5:
            training_kspace = kspace * training_mask
            validation_kspace = kspace * validation_mask
        elif kspace.ndim == 6:
            training_kspace = kspace * training_mask.unsqueeze(1)
            validation_kspace = kspace * validation_mask.unsqueeze(1)
        else:
            raise ValueError(f"Unexpected number of kspace dimensions: {kspace.ndim}")

        return training_mask, training_kspace, validation_mask, validation_kspace


def compute_ttv(reconstruction):
    """
    Compute the temporal total variation (TV) of the reconstructed video.
    """
    assert reconstruction.ndim == 3, f"Expected reconstruction to have 3 dimensions (complex-valued: f, h, w), got {reconstruction.ndim}"

    # Compute temporal differences on magnitude
    reconstruction = torch.abs(reconstruction) 
    temporal_diff = reconstruction[1:] - reconstruction[:-1]

    # Compute the L1 norm of the temporal differences
    tv = torch.sum(torch.abs(temporal_diff))

    # Normalize by the reconstruction size to get an average TV per pixel
    tv = tv / np.prod(reconstruction.shape)

    return tv.item()

