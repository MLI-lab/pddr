"""
Based on matlab code provided by the CMRxRecon challenge:
  https://github.com/CmrxRecon/CMRxRecon2024/tree/main/CMRxReconMaskGeneration/Task2/Toolbox_Mask_Generator_Task2
  If you want to use the code, please cite the following paper:
  Zi Wang et al., Deep Separable Spatiotemporal Learning for Fast Dynamic Cardiac MRI, arXiv:2402.15939, 2024.
"""

import numpy as np

from math import ceil, floor
from scipy.linalg import toeplitz
from scipy.ndimage import rotate

def zpad(x, sx, sy, sz=None, st=None):
    s = [sx, sy]
    if sz is not None:
        s.append(sz)
    if st is not None:
        s.append(st)
    
    m = list(x.shape)
    if len(m) < len(s):
        m.extend([1] * (len(s) - len(m)))
    
    if m == s:
        return x
    
    res = np.zeros(s, dtype=x.dtype)
    idx = [slice(floor(s[i]/2) + ceil(-m[i]/2), floor(s[i]/2) + ceil(m[i]/2)) for i in range(len(s))]
    res[tuple(idx)] = x
    return res

def imrotate(image, angle, mode='constant'):
    return rotate(image, angle, reshape=False, mode=mode)

def crop(x, sx, sy):
    cx, cy = x.shape[0] // 2, x.shape[1] // 2
    return x[cx - sx // 2:cx + sx // 2, cy - sy // 2:cy + sy // 2]

def ktdup(ph, ti, ny, nt):
    ph = ph + ceil((ny + 1) / 2)
    ti = ti + ceil((nt + 1) / 2)
    pt = (ti - 1) * ny + ph

    uniquept, countOfpt = np.unique(pt, return_counts=True)
    repeatedValues = uniquept[countOfpt != 1]

    if len(repeatedValues) == 0:
        # No duplicates, return original values
        ph = ph - ceil((ny + 1) / 2)
        ti = ti - ceil((nt + 1) / 2)
        return ph, ti
    
    dupind = np.concatenate([np.where(pt == val)[0][1:] for val in repeatedValues])
    empind = np.setdiff1d(np.arange(1, ny * nt + 1), pt)

    for i in dupind:
        newind = nearestvac(pt[i], empind, ny)
        pt[i] = newind
        empind = np.setdiff1d(empind, newind)

    ph, ti = ind2xy(pt, ny)
    ph = ph - ceil((ny + 1) / 2)
    ti = ti - ceil((nt + 1) / 2)
    return ph, ti

def nearestvac(dupind, empind, ny):
    x0, y0 = ind2xy(dupind, ny)
    x, y = ind2xy(empind, ny)
    dis = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
    return empind[np.argmin(dis)]

def ind2xy(ind, X):
    x = ind - (ind - 1) // X * X
    y = np.ceil(ind / X).astype(int)
    return x, y

def randp(P, rng, *args):
    """Sample indices from a discrete probability distribution.
    
    Args:
        P: Probability vector.
        rng: np.random.RandomState instance (or int seed for backward compatibility).
        *args: Shape arguments passed to rng.rand().
    """
    if isinstance(rng, (int, np.integer)):
        rng = np.random.RandomState(int(rng))
    X = rng.rand(*args)
    P = P.flatten()
    if np.any(P < 0):
        raise ValueError('All probabilities should be 0 or larger.')
    if np.sum(P) == 0:
        X.fill(0)
    else:
        # X = np.histogram(X, bins=np.concatenate(([0], np.cumsum(P) / np.sum(P))))[0]
        cum_prob = np.cumsum(P) / np.sum(P)
        X = np.searchsorted(cum_prob, X.flatten()).reshape(X.shape)
    return X

def ktUniformSampling(nx, ny, nt, ncalib, R):
    ptmp = np.zeros(ny)
    ptmp[np.round(np.arange(0, ny, R))] = 1
    ttmp = np.zeros(nt)
    ttmp[np.round(np.arange(0, nt , R))] = 1
    Top = toeplitz(ptmp, ttmp)
    ind = np.where(Top)
    ph = ind[0] - floor(ny / 2)
    ti = ind[1] - floor(nt / 2)

    ph, ti = ktdup(ph, ti, ny, nt)
    samp = np.zeros((ny, nt), dtype=int)
    # ind = np.round(ny * (ti + floor(nt / 2)) + (ph + floor(ny / 2) + 1)).astype(int)
    ind = ((ph + floor(ny / 2)).astype(int), (ti + floor(nt / 2)).astype(int))
    # ind[ind <= 0] = 1
    samp[ind] = 1

    acs = zpad(np.ones((nx, ncalib, nt)), nx, ny, nt)
    ktus = np.transpose(np.tile(samp, (nx, 1, 1)), (0, 1, 2))
    mask_temp = ktus + acs
    mask = (mask_temp > 0).astype(int)
    return mask

def ktGaussianSampling(nx, ny, nt, ncalib, R, alpha, seed):
    p1 = np.arange(-floor(ny / 2), ceil(ny / 2))
    tr = round(ny / R)
    ti = np.full(tr * nt, 999999999, dtype=int)
    ph = np.zeros(tr * nt, dtype=int)
    sig = ny / 5
    prob = 0.1 + alpha / (1 - alpha + 1e-10) * np.exp(-(p1 ** 2) / (1 * sig ** 2))
    rng = np.random.RandomState(seed)
    tmpSd = np.round(1e6 * rng.rand(nt)).astype(int)
    ind = 0
    for i in range(-floor(nt / 2), ceil(nt / 2)):
        a = np.where(ti == i)[0]
        n_tmp = tr - len(a)
        prob_tmp = prob.copy()
        prob_tmp[a] = 0
        frame_rng = np.random.RandomState(int(tmpSd[i + floor(nt / 2)]))
        p_tmp = randp(prob_tmp, frame_rng, n_tmp)
        ti[ind:ind + n_tmp] = i
        ph[ind:ind + n_tmp] = p_tmp - floor(ny / 2) - 1
        ind += n_tmp

    ph, ti = ktdup(ph, ti, ny, nt)
    samp = np.zeros((ny, nt), dtype=int)
    # ind = np.round(ny * (ti + floor(nt / 2)) + (ph + floor(ny / 2))).astype(int)
    ind = ((ph + floor(ny / 2)).astype(int), (ti + floor(nt / 2)).astype(int))
    samp[ind] = 1

    acs = zpad(np.ones((nx, ncalib, nt)), nx, ny, nt)
    ktus = np.transpose(np.tile(samp, (nx, 1, 1)), (0, 1, 2))
    mask_temp = ktus + acs
    mask = (mask_temp > 0).astype(int)
    return mask

def ktRadialSampling(nx, ny, nt, ncalib, R, angle4next, cropcorner):
    rate = 1 / R
    beams = floor(rate * 180)
    a = max(nx, ny) if cropcorner else ceil(np.sqrt(2) * max(nx, ny))
    aux = np.zeros((a, a))
    aux[round(a / 2), :] = 1
    angle = 180 / beams

    ktus = np.zeros((nx, ny, nt))
    for i in range(nt):
        angles = np.arange(0 + angle4next * i, 180 + angle4next * i, angle)
        image = np.zeros((nx, ny))
        for ang in angles:
            temp = crop(imrotate(aux, ang, 'constant'), nx, ny)
            image += (temp > 0.5)
            image = np.clip(image, 0, 1)
        ktus[:, :, i] = image

    acs = zpad(np.ones((ncalib, ncalib, nt)), nx, ny, nt)
    mask_temp = ktus + acs
    mask = (mask_temp > 0).astype(int)
    return mask

def ktMaskGenerator(nx, ny, nt, ncalib, R, pattern, alpha=0.28, seed=1996):
    if pattern == 'ktUniform':
        return ktUniformSampling(nx, ny, nt, ncalib, R)
    elif pattern == 'ktGaussian':
        # alpha = 0.28 # default: 0.28, 0<alpha<1 controls sampling density; 0: uniform density, 1: maximally non-uniform density (Gaussian)
        return ktGaussianSampling(nx, ny, nt, ncalib, R, alpha, seed)
    elif pattern == 'ktRadial':
        angle4next = 137.5
        cropcorner = True
        R = R * 0.6
        return ktRadialSampling(nx, ny, nt, ncalib, R, angle4next, cropcorner)
    else:
        raise ValueError('No selected undersampling pattern. Please choose the proper one.')


"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import contextlib
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch


@contextlib.contextmanager
def temp_seed(rng: np.random.RandomState, seed: Optional[Union[int, Tuple[int, ...]]]):
    """A context manager for temporarily adjusting the random seed."""
    if seed is None:
        try:
            yield
        finally:
            pass
    else:
        state = rng.get_state()
        rng.seed(seed)
        try:
            yield
        finally:
            rng.set_state(state)


class MaskFunc:
    """
    An object for GRAPPA-style sampling masks.

    This crates a sampling mask that densely samples the center while
    subsampling outer k-space regions based on the undersampling factor.

    When called, ``MaskFunc`` uses internal functions create mask by 1)
    creating a mask for the k-space center, 2) create a mask outside of the
    k-space center, and 3) combining them into a total mask. The internals are
    handled by ``sample_mask``, which calls ``calculate_center_mask`` for (1)
    and ``calculate_acceleration_mask`` for (2). The combination is executed
    in the ``MaskFunc`` ``__call__`` function.

    If you would like to implement a new mask, simply subclass ``MaskFunc``
    and overwrite the ``sample_mask`` logic. See examples in ``RandomMaskFunc``
    and ``EquispacedMaskFunc``.
    """

    def __init__(
        self,
        center_fractions: Sequence[float],
        accelerations: Sequence[int],
        allow_any_combination: bool = False,
        seed: Optional[int] = None,
    ):
        """
        Args:
            center_fractions: Fraction of low-frequency columns to be retained.
                If multiple values are provided, then one of these numbers is
                chosen uniformly each time.
            accelerations: Amount of under-sampling. This should have the same
                length as center_fractions. If multiple values are provided,
                then one of these is chosen uniformly each time.
            allow_any_combination: Whether to allow cross combinations of
                elements from ``center_fractions`` and ``accelerations``.
            seed: Seed for starting the internal random number generator of the
                ``MaskFunc``.
        """
        if len(center_fractions) != len(accelerations) and not allow_any_combination:
            raise ValueError(
                "Number of center fractions should match number of accelerations "
                "if allow_any_combination is False."
            )

        self.center_fractions = center_fractions
        self.accelerations = accelerations
        self.allow_any_combination = allow_any_combination
        self.rng = np.random.RandomState(seed)

    def __call__(
        self,
        shape: Sequence[int],
        offset: Optional[int] = None,
        seed: Optional[Union[int, Tuple[int, ...]]] = None,
    ) -> Tuple[torch.Tensor, int]:
        """
        Sample and return a k-space mask.

        Args:
            shape: Shape of k-space.
            offset: Offset from 0 to begin mask (for equispaced masks). If no
                offset is given, then one is selected randomly.
            seed: Seed for random number generator for reproducibility.

        Returns:
            A 2-tuple containing 1) the k-space mask and 2) the number of
            center frequency lines.
        """
        if len(shape) < 3:
            raise ValueError("Shape should have 3 or more dimensions")

        with temp_seed(self.rng, seed):
            center_mask, accel_mask, num_low_frequencies = self.sample_mask(
                shape, offset
            )

        # combine masks together
        return torch.max(center_mask, accel_mask), num_low_frequencies

    def sample_mask(
        self,
        shape: Sequence[int],
        offset: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Sample a new k-space mask.

        This function samples and returns two components of a k-space mask: 1)
        the center mask (e.g., for sensitivity map calculation) and 2) the
        acceleration mask (for the edge of k-space). Both of these masks, as
        well as the integer of low frequency samples, are returned.

        Args:
            shape: Shape of the k-space to subsample.
            offset: Offset from 0 to begin mask (for equispaced masks).

        Returns:
            A 3-tuple contaiing 1) the mask for the center of k-space, 2) the
            mask for the high frequencies of k-space, and 3) the integer count
            of low frequency samples.
        """
        num_cols = shape[-2]
        center_fraction, acceleration = self.choose_acceleration()
        num_low_frequencies = round(num_cols * center_fraction)
        center_mask = self.reshape_mask(
            self.calculate_center_mask(shape, num_low_frequencies), shape
        )
        acceleration_mask = self.reshape_mask(
            self.calculate_acceleration_mask(
                num_cols, acceleration, offset, num_low_frequencies
            ),
            shape,
        )

        return center_mask, acceleration_mask, num_low_frequencies

    def reshape_mask(self, mask: np.ndarray, shape: Sequence[int]) -> torch.Tensor:
        """Reshape mask to desired output shape."""
        num_cols = shape[-2]
        mask_shape = [1 for _ in shape]
        mask_shape[-2] = num_cols

        return torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32))

    def calculate_acceleration_mask(
        self,
        num_cols: int,
        acceleration: int,
        offset: Optional[int],
        num_low_frequencies: int,
    ) -> np.ndarray:
        """
        Produce mask for non-central acceleration lines.

        Args:
            num_cols: Number of columns of k-space (2D subsampling).
            acceleration: Desired acceleration rate.
            offset: Offset from 0 to begin masking (for equispaced masks).
            num_low_frequencies: Integer count of low-frequency lines sampled.

        Returns:
            A mask for the high spatial frequencies of k-space.
        """
        raise NotImplementedError

    def calculate_center_mask(
        self, shape: Sequence[int], num_low_freqs: int
    ) -> np.ndarray:
        """
        Build center mask based on number of low frequencies.

        Args:
            shape: Shape of k-space to mask.
            num_low_freqs: Number of low-frequency lines to sample.

        Returns:
            A mask for hte low spatial frequencies of k-space.
        """
        num_cols = shape[-2]
        mask = np.zeros(num_cols, dtype=np.float32)
        pad = (num_cols - num_low_freqs + 1) // 2
        mask[pad : pad + num_low_freqs] = 1
        assert mask.sum() == num_low_freqs

        return mask

    def choose_acceleration(self):
        """Choose acceleration based on class parameters."""
        if self.allow_any_combination:
            return self.rng.choice(self.center_fractions), self.rng.choice(
                self.accelerations
            )
        else:
            choice = self.rng.randint(len(self.center_fractions))
            return self.center_fractions[choice], self.accelerations[choice]


class Random2dMaskFunc(MaskFunc):
    """
    Creates a random sub-sampling mask of a given shape.

    The mask selects a subset of indicies from the input k-space data. If the
    k-space data has N indicies, the mask picks out:
        1. N_low_freqs = (N * center_fraction) indicies in the center
           corresponding to low-frequencies.
        2. The other indicies are selected uniformly at random with a
        probability equal to: prob = (N / acceleration - N_low_freqs) /
        (N - N_low_freqs). This ensures that the expected number of indicies
        selected is equal to (N / acceleration).

    It is possible to use multiple center_fractions and accelerations, in which
    case one possible (center_fraction, acceleration) is chosen uniformly at
    random each time the ``RandomMaskFunc`` object is called.

    For example, if accelerations = [4, 8] and center_fractions = [0.08, 0.04],
    then there is a 50% probability that 4-fold acceleration with 8% center
    fraction is selected and a 50% probability that 8-fold acceleration with 4%
    center fraction is selected.
    """

    def sample_mask(
        self,
        shape: Sequence[int],
        offset: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        
        assert shape[0] == 1  
        fe, pe = shape[1:]

        center_fraction, acceleration = self.choose_acceleration()
        center_size = min(fe, pe)
        num_low_frequencies = round(center_size * center_fraction)

        center_mask = torch.zeros((fe, pe))
        center_mask[fe//2 - num_low_frequencies//2:fe//2 + num_low_frequencies//2, pe//2 - num_low_frequencies//2:pe//2 + num_low_frequencies//2] = 1
        center_mask = center_mask.unsqueeze(0)

        acceleration_mask = torch.zeros((fe, pe))
        num_selected = int(1/acceleration * fe * pe)
        indices = torch.randperm(fe * pe)[:num_selected]
        acceleration_mask.view(-1)[indices] = 1
        acceleration_mask = acceleration_mask.unsqueeze(0)

        return center_mask, acceleration_mask, num_low_frequencies
    

class RandomMaskFunc(MaskFunc):
    """
    Creates a random sub-sampling mask of a given shape.

    The mask selects a subset of columns from the input k-space data. If the
    k-space data has N columns, the mask picks out:
        1. N_low_freqs = (N * center_fraction) columns in the center
           corresponding to low-frequencies.
        2. The other columns are selected uniformly at random with a
        probability equal to: prob = (N / acceleration - N_low_freqs) /
        (N - N_low_freqs). This ensures that the expected number of columns
        selected is equal to (N / acceleration).

    It is possible to use multiple center_fractions and accelerations, in which
    case one possible (center_fraction, acceleration) is chosen uniformly at
    random each time the ``RandomMaskFunc`` object is called.

    For example, if accelerations = [4, 8] and center_fractions = [0.08, 0.04],
    then there is a 50% probability that 4-fold acceleration with 8% center
    fraction is selected and a 50% probability that 8-fold acceleration with 4%
    center fraction is selected.
    """

    def calculate_acceleration_mask(
        self,
        num_cols: int,
        acceleration: int,
        offset: Optional[int],
        num_low_frequencies: int,
    ) -> np.ndarray:
        prob = (num_cols / acceleration - num_low_frequencies) / (
            num_cols - num_low_frequencies
        )

        return self.rng.uniform(size=num_cols) < prob


class EquiSpacedMaskFunc(MaskFunc):
    """
    Sample data with equally-spaced k-space lines.

    The lines are spaced exactly evenly, as is done in standard GRAPPA-style
    acquisitions. This means that with a densely-sampled center,
    ``acceleration`` will be greater than the true acceleration rate.
    """

    def calculate_acceleration_mask(
        self,
        num_cols: int,
        acceleration: int,
        offset: Optional[int],
        num_low_frequencies: int,
    ) -> np.ndarray:
        """
        Produce mask for non-central acceleration lines.

        Args:
            num_cols: Number of columns of k-space (2D subsampling).
            acceleration: Desired acceleration rate.
            offset: Offset from 0 to begin masking. If no offset is specified,
                then one is selected randomly.
            num_low_frequencies: Not used.

        Returns:
            A mask for the high spatial frequencies of k-space.
        """
        if offset is None:
            offset = self.rng.randint(0, high=round(acceleration))

        mask = np.zeros(num_cols, dtype=np.float32)
        mask[offset::acceleration] = 1

        return mask


class EquispacedMaskFractionFunc(MaskFunc):
    """
    Equispaced mask with approximate acceleration matching.

    The mask selects a subset of columns from the input k-space data. If the
    k-space data has N columns, the mask picks out:
        1. N_low_freqs = (N * center_fraction) columns in the center
           corresponding to low-frequencies.
        2. The other columns are selected with equal spacing at a proportion
           that reaches the desired acceleration rate taking into consideration
           the number of low frequencies. This ensures that the expected number
           of columns selected is equal to (N / acceleration)

    It is possible to use multiple center_fractions and accelerations, in which
    case one possible (center_fraction, acceleration) is chosen uniformly at
    random each time the EquispacedMaskFunc object is called.

    Note that this function may not give equispaced samples (documented in
    https://github.com/facebookresearch/fastMRI/issues/54), which will require
    modifications to standard GRAPPA approaches. Nonetheless, this aspect of
    the function has been preserved to match the public multicoil data.
    """

    def calculate_acceleration_mask(
        self,
        num_cols: int,
        acceleration: int,
        offset: Optional[int],
        num_low_frequencies: int,
    ) -> np.ndarray:
        """
        Produce mask for non-central acceleration lines.

        Args:
            num_cols: Number of columns of k-space (2D subsampling).
            acceleration: Desired acceleration rate.
            offset: Offset from 0 to begin masking. If no offset is specified,
                then one is selected randomly.
            num_low_frequencies: Number of low frequencies. Used to adjust mask
                to exactly match the target acceleration.

        Returns:
            A mask for the high spatial frequencies of k-space.
        """
        # determine acceleration rate by adjusting for the number of low frequencies
        adjusted_accel = (acceleration * (num_low_frequencies - num_cols)) / (
            num_low_frequencies * acceleration - num_cols
        )
        if offset is None:
            offset = self.rng.randint(0, high=round(adjusted_accel))

        mask = np.zeros(num_cols)
        accel_samples = np.arange(offset, num_cols - 1, adjusted_accel)
        accel_samples = np.around(accel_samples).astype(np.uint)
        mask[accel_samples] = 1.0

        return mask


class MagicMaskFunc(MaskFunc):
    """
    Masking function for exploiting conjugate symmetry via offset-sampling.

    This function applies the mask described in the following paper:

    Defazio, A. (2019). Offset Sampling Improves Deep Learning based
    Accelerated MRI Reconstructions by Exploiting Symmetry. arXiv preprint,
    arXiv:1912.01101.

    It is essentially an equispaced mask with an offset for the opposite site
    of k-space. Since MRI images often exhibit approximate conjugate k-space
    symmetry, this mask is generally more efficient than a standard equispaced
    mask.

    Similarly to ``EquispacedMaskFunc``, this mask will usually undereshoot the
    target acceleration rate.
    """

    def calculate_acceleration_mask(
        self,
        num_cols: int,
        acceleration: int,
        offset: Optional[int],
        num_low_frequencies: int,
    ) -> np.ndarray:
        """
        Produce mask for non-central acceleration lines.

        Args:
            num_cols: Number of columns of k-space (2D subsampling).
            acceleration: Desired acceleration rate.
            offset: Offset from 0 to begin masking. If no offset is specified,
                then one is selected randomly.
            num_low_frequencies: Not used.

        Returns:
            A mask for the high spatial frequencies of k-space.
        """
        if offset is None:
            offset = self.rng.randint(0, high=acceleration)

        if offset % 2 == 0:
            offset_pos = offset + 1
            offset_neg = offset + 2
        else:
            offset_pos = offset - 1 + 3
            offset_neg = offset - 1 + 0

        poslen = (num_cols + 1) // 2
        neglen = num_cols - (num_cols + 1) // 2
        mask_positive = np.zeros(poslen, dtype=np.float32)
        mask_negative = np.zeros(neglen, dtype=np.float32)

        mask_positive[offset_pos::acceleration] = 1
        mask_negative[offset_neg::acceleration] = 1
        mask_negative = np.flip(mask_negative)

        mask = np.concatenate((mask_positive, mask_negative))

        return np.fft.fftshift(mask)  # shift mask and return


class MagicMaskFractionFunc(MagicMaskFunc):
    """
    Masking function for exploiting conjugate symmetry via offset-sampling.

    This function applies the mask described in the following paper:

    Defazio, A. (2019). Offset Sampling Improves Deep Learning based
    Accelerated MRI Reconstructions by Exploiting Symmetry. arXiv preprint,
    arXiv:1912.01101.

    It is essentially an equispaced mask with an offset for the opposite site
    of k-space. Since MRI images often exhibit approximate conjugate k-space
    symmetry, this mask is generally more efficient than a standard equispaced
    mask.

    Similarly to ``EquispacedMaskFractionFunc``, this method exactly matches
    the target acceleration by adjusting the offsets.
    """

    def sample_mask(
        self,
        shape: Sequence[int],
        offset: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Sample a new k-space mask.

        This function samples and returns two components of a k-space mask: 1)
        the center mask (e.g., for sensitivity map calculation) and 2) the
        acceleration mask (for the edge of k-space). Both of these masks, as
        well as the integer of low frequency samples, are returned.

        Args:
            shape: Shape of the k-space to subsample.
            offset: Offset from 0 to begin mask (for equispaced masks).

        Returns:
            A 3-tuple contaiing 1) the mask for the center of k-space, 2) the
            mask for the high frequencies of k-space, and 3) the integer count
            of low frequency samples.
        """
        num_cols = shape[-2]
        fraction_low_freqs, acceleration = self.choose_acceleration()
        num_cols = shape[-2]
        num_low_frequencies = round(num_cols * fraction_low_freqs)

        # bound the number of low frequencies between 1 and target columns
        target_columns_to_sample = round(num_cols / acceleration)
        num_low_frequencies = max(min(num_low_frequencies, target_columns_to_sample), 1)

        # adjust acceleration rate based on target acceleration.
        adjusted_target_columns_to_sample = (
            target_columns_to_sample - num_low_frequencies
        )
        adjusted_acceleration = 0
        if adjusted_target_columns_to_sample > 0:
            adjusted_acceleration = round(num_cols / adjusted_target_columns_to_sample)

        center_mask = self.reshape_mask(
            self.calculate_center_mask(shape, num_low_frequencies), shape
        )
        accel_mask = self.reshape_mask(
            self.calculate_acceleration_mask(
                num_cols, adjusted_acceleration, offset, num_low_frequencies
            ),
            shape,
        )

        return center_mask, accel_mask, num_low_frequencies


def create_mask_for_mask_type(
    mask_type_str: str,
    center_fractions: Sequence[float],
    accelerations: Sequence[int],
) -> MaskFunc:
    """
    Creates a mask of the specified type.

    Args:
        center_fractions: What fraction of the center of k-space to include.
        accelerations: What accelerations to apply.

    Returns:
        A mask func for the target mask type.
    """
    if mask_type_str == "random":
        mask_func = RandomMaskFunc(center_fractions, accelerations)
        # return lambda shape, seed: mask_func((1, shape[-1], 1), seed=seed)[0].permute(0, 2, 1).repeat(1, shape[-2], 1)
        return lambda shape, seed: mask_func((1, shape[-2], 1), seed=seed)[0].repeat(1, 1, shape[-1])
    elif mask_type_str == "equispaced":
        mask_func =  EquiSpacedMaskFunc(center_fractions, accelerations)
        return lambda shape, seed: mask_func((1, shape[-2], 1), seed=seed)[0].repeat(1, 1, shape[-1])
    elif mask_type_str == "equispaced_fraction":
        mask_func =  EquispacedMaskFractionFunc(center_fractions, accelerations)
        return lambda shape, seed: mask_func((1, shape[-2], 1), seed=seed)[0].repeat(1, 1, shape[-1])
    # elif mask_type_str == "magic":
    #     return MagicMaskFunc(center_fractions, accelerations)
    # elif mask_type_str == "magic_fraction":
    #     return MagicMaskFractionFunc(center_fractions, accelerations)
    elif mask_type_str == "random_2d":
        mask_func =  Random2dMaskFunc(center_fractions, accelerations)
        return lambda shape, seed: mask_func(shape, seed=seed)[0]
    else:
        raise ValueError(f"{mask_type_str} not supported")



# # Example usage
# if __name__ == "__main__":
#     from matplotlib import pyplot as plt

#     nx = 264
#     ny = 186
#     nt = 120
#     ncalib = 0
#     R = 12
#     alpha = 0.3

#     for pattern in ['ktGaussian']: #['ktUniform', 'ktGaussian', 'ktRadial']:
#         mask = ktMaskGenerator(nx, ny, nt, ncalib, R, pattern, alpha)
#         mask = mask.transpose((2, 1, 0)) # nt, ny, nx
        
#         # Mask display
#         print(f'Actual AccFactor: {1/((mask != 0).mean())}')
#         plt.figure()
#         plt.imshow(mask[0, :, :], cmap='gray')
#         plt.axis('off')
#         plt.figure()
#         plt.imshow(mask[:, :, 60], cmap='gray')
#         plt.axis('off')
#         plt.show()
