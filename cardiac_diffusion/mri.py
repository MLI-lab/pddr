import torch


def fftnc(data: torch.Tensor, norm='ortho', dim=(1, 2)) -> torch.Tensor:
    """
    Apply centered n-dimensional Fast Fourier Transform.

    Parameters:
        data: Complex valued input data
        norm: Normalization mode
        dim: dimensions to be shifted (i.e. `ndim`-dimensional Fourier shift)

    Returns:
        The FFT of the input.
    """
    data = torch.fft.ifftshift(data, dim=dim)
    data = torch.fft.fftn(data, norm=norm, dim=dim)
    data = torch.fft.fftshift(data, dim=dim)

    return data

def ifftnc(data: torch.Tensor, norm='ortho', dim=(1, 2)) -> torch.Tensor:
    """
    Apply centered n-dimensional Inverse Fast Fourier Transform.

    Parameters:
        data: Complex valued input data
        norm: Normalization mode
        dim: dimensions to be shifted (i.e. `ndim`-dimensional Fourier shift)

    Returns:
        The IFFT of the input.
    """
    data = torch.fft.ifftshift(data, dim=dim)
    data = torch.fft.ifftn(data, norm=norm, dim=dim)
    data = torch.fft.fftshift(data, dim=dim)

    return data

def rss(data: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """
    Compute the Root Sum of Squares (RSS).

    Args:
        data: The complex input tensor.
        dim: The dimensions along which to apply the RSS transform (coil dimension).

    Returns:
        The RSS image.
    """

    # TODO: Check if this is correct wrt complex or real image
    return torch.sqrt((data ** 2).sum(dim))

def adaptive_combine(data: torch.Tensor, sens_maps: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """
    Compute a MVUE image by combining coil images with coil sensitivities.

    Args:
        data: The individual coil images.
        coil_sens: The coil sensitivity maps.
        dim: The dimension along which to combine the images (coil dimension).

    Returns:
        The MVUE image.
    """
    return (data * torch.conj(sens_maps)).sum(dim)

def image2kspace(image: torch.Tensor, sens_maps: torch.Tensor, norm='ortho', dim=(1, 2)) -> torch.Tensor:
    """
    Expand the image to multiple sensitivity weighted coil images, and compute the Fourier transform of these.

    Args:
        image: The complex input image.
        sens_maps: The coil sensitivity maps.
        norm: The normalization mode.
        dim: The dimensions to be shifted (i.e. `ndim`-dimensional Fourier shift).

    Returns:
        The k-space representation of the input image.
    """
    return fftnc(image*sens_maps, norm=norm, dim=dim)

def kspace2image(kspace: torch.Tensor, norm='ortho', fdim=(1, 2), cdim=0, adaptive=True, sens_maps: torch.Tensor=None) -> torch.Tensor:
    """
    Compute the inverse Fourier transform of the k-space data and combine these coil images to one image.

    Args:
        kspace: The input k-space data.
        norm: The normalization mode.
        fdim: The dimensions to be shifted (i.e. `ndim`-dimensional Fourier shift).
        cdim: The coil dimension.
        adaptive: Whether to use adaptive coil combination or RSS.
        sens_maps: The coil sensitivity maps.
    Returns:
        The image representation of the input k-space data.
    """
    coil_images = ifftnc(kspace, norm=norm, dim=fdim)

    if adaptive:
        return adaptive_combine(coil_images, sens_maps, dim=cdim)
    else:
        return rss(coil_images, dim=cdim)
