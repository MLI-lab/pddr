import torch
import numpy as np
import h5py
import json

from pathlib import Path
from torch.utils import data
from torchvision.transforms import CenterCrop

from cardiac_diffusion.masking import ktMaskGenerator
from cardiac_diffusion.mri import image2kspace

class CMRxProcessed(data.Dataset):
    def __init__(
            self, 
            data_path, 
            repeat_cycle=0,
            undersample=False, 
            undersampling_factor=12, 
            undersampling_type='ktGaussian',
            alpha=0.28,
            seed=1996,
            filter=None, 
            multicoil=True, 
            image_only=False, 
            crop_size=(246, 512),
            return_fname=False
        ):
        super().__init__()
        self.data_path = data_path
        self.image_only = image_only
        self.crop_size = crop_size
        self.repeat_cycle = repeat_cycle
        self.multicoil = multicoil
        self.undersampling_factor = undersampling_factor
        self.undersample = undersample
        self.undersampling_type = undersampling_type
        self.alpha = alpha
        self.seed = seed
        self.return_fname = return_fname
        filter = '' if filter is None else filter
        if isinstance(self.undersampling_factor, list) or undersample:
            self.mask_key = 'mask'
        else:
            self.mask_key = f"mask{undersampling_factor:02}"

        # find all files
        if isinstance(data_path, (list, tuple)):
            paths = []
            for single_dir in data_path:
                single_dir = Path(single_dir)
                if single_dir.is_dir():
                    paths.extend(single_dir.glob(f'**/*{filter}.h5'))
                else:
                    print(f"[Warning] {single_dir} is not a valid directory. Skipping.")
        else:
            p = Path(data_path)
            if p.is_dir():
                paths = p.glob(f'**/*{filter}.h5')
            elif p.is_file():
                paths = [p]
            else:
                raise ValueError(f"{data_path} is not a valid directory or file.")

        # split each file to return slices individually
        self.raw_samples = []
        for fname in paths:
            with h5py.File(fname, "r") as hf:
                num_slices = hf['kspace'].shape[1]  # count slices in [t, s, ky, kx] or [t, s, coil, ky, kx]  

            new_raw_samples = []
            for slice_ind in range(num_slices):
                raw_sample = (fname, slice_ind)
                new_raw_samples.append(raw_sample)
            self.raw_samples += new_raw_samples

    def _normalize(self, x, norm):
        # # normalize by the kspace shape (image-dims + frames), but not by the number of coils, as this is done implicitly via sensitivity maps
        x = x * np.sqrt(2*x.shape[0]*np.prod(x.shape[-2:])) / norm
        return x

    def __len__(self):
        return len(self.raw_samples)

    def __getitem__(self, idx):
        fname, dataslice = self.raw_samples[idx]

        with h5py.File(fname, "r") as hf:
            kspace = hf['kspace'][:, dataslice]                                                                # [frame, coil, ky, kx] or [frame, ky, kx]
            ground = hf['image'][:, dataslice] if 'image' in hf else None                                      # [frame, ky, kx]
            sensitivities = hf['sensitivities'][dataslice] if self.multicoil and not self.image_only else None # [coil, ky, kx]
            if self.mask_key in hf:
                mask = hf[self.mask_key][:] if not self.undersample and not self.image_only else None          # [ky, kx] or [frame, slice, ky, kx]
            else:
                mask = hf['mask'][:] if not self.undersample and not self.image_only else None                 # [ky, kx] or [frame, slice, ky, kx]
            mask = mask[:, dataslice, None]  if mask is not None and mask.ndim == 4 else mask                  # [frame, 1, ky, kx]
            
        kspace = torch.from_numpy(kspace)
        ground = torch.from_numpy(ground) if ground is not None else None
        sensitivities = torch.from_numpy(sensitivities) if sensitivities is not None else None
        mask = torch.from_numpy(mask) if mask is not None else None

        ground = self._normalize(ground, kspace.norm()) if ground is not None else None

        if self.repeat_cycle > 1:
            kspace = kspace.repeat(self.repeat_cycle, *[1]*(kspace.ndim-1))  # repeat along the temporal dimension
            if ground is not None:
                ground = ground.repeat(self.repeat_cycle, *[1]*(ground.ndim-1))
            if (not self.undersample) and (mask is not None):
                mask = mask.repeat(self.repeat_cycle, *[1]*(mask.ndim-1))

        if self.image_only:
            ground = torch.stack((ground.real, ground.imag), dim=0)
            crop = CenterCrop(self.crop_size)
            return crop(ground)
        
        if self.undersample:
            nt, ny, nx = ground.shape
            ncalib = 0
            
            if isinstance(self.undersampling_factor, list):
                undersampling_f = np.random.choice(self.undersampling_factor)
            else:
                undersampling_f = self.undersampling_factor

            # Per-sample seed: each example gets a unique but deterministic mask
            sample_seed = self.seed + idx
            mask = ktMaskGenerator(nx, ny, nt, ncalib, undersampling_f, self.undersampling_type, self.alpha, sample_seed)
            mask = mask.transpose((2, 1, 0))
            mask = torch.from_numpy(mask)
            mask = mask.unsqueeze(1)            # [frame, 1, ky, kx]
        
        # masking, transformation, normalization, and make complex dim explicit [frame, coil, ky, kx] -> [2, frame, coil, ky, kx]
        # fair normalization setting, as typically one would not have access to the fully sampled kspace
        masked_kspace = kspace * mask
        norm_factor = np.sqrt(self.undersampling_factor/2.0) # empirical normalization factor to account for undersampling
        masked_kspace = self._normalize(masked_kspace, masked_kspace.norm()*norm_factor)
        masked_kspace = torch.stack((masked_kspace.real, masked_kspace.imag), dim=0)

        if self.return_fname:
            name = f'{fname.parent.stem}_{dataslice}'
            return masked_kspace, mask, sensitivities, ground, name
        
        return masked_kspace, mask, sensitivities, ground


class CMRxTestset(data.Dataset):
    def __init__(
            self, 
            data_path, 
            repeat_cycle=0,
            undersample=True, 
            undersampling_factor=12, 
            undersampling_type='ktGaussian',
            alpha=0.28,
            seed=1996
        ):
        super().__init__()
        self.data_path = data_path
        self.repeat_cycle = repeat_cycle
        self.undersample = undersample
        self.undersampling_factor = undersampling_factor
        self.undersampling_type = undersampling_type
        self.alpha = alpha
        self.seed = seed

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
            # p = Path(data_path)
            # if p.is_dir():
            #     paths = p.glob(f'**/*.h5')
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


    def _normalize(self, x, norm):
        x = x * np.sqrt(2*x.shape[0]*np.prod(x.shape[-2:])) / norm
        return x

    def __len__(self):
        return len(self.raw_samples)

    def __getitem__(self, idx):
        fname, dataslice = self.raw_samples[idx]

        with h5py.File(fname, "r") as hf:
            kspace = hf['kspace'][:, dataslice]                                      # [frame, coil, ky, kx] or [frame, ky, kx]
            ground = hf['image'][:, dataslice]                                       # [frame, ky, kx]
            sensitivities = hf['sensitivities'][dataslice]                           # [coil, ky, kx]
            
        kspace = torch.from_numpy(kspace)
        ground = torch.from_numpy(ground) if ground is not None else None
        sensitivities = torch.from_numpy(sensitivities) if sensitivities is not None else None

        if self.repeat_cycle > 1:
            # repeat along the temporal dimension
            if ground is not None:
                ground = ground.repeat(self.repeat_cycle, *[1]*(ground.ndim-1))
                # kspace = kspace.repeat(self.repeat_cycle, *[1]*(kspace.ndim-1)) 
                kspace = image2kspace(ground.unsqueeze(1), sensitivities.unsqueeze(0), dim=(2, 3))
            else:
                raise ValueError("Ground truth must be provided for repeating the cycle.")
        
        ground = self._normalize(ground, kspace.norm()) if ground is not None else None
        
        if self.undersample:
            nt, ny, nx = ground.shape
            ncalib = 0

            # Per-sample seed: each example gets a unique but deterministic mask
            sample_seed = self.seed + idx
            mask = ktMaskGenerator(nx, ny, nt, ncalib, self.undersampling_factor, self.undersampling_type, self.alpha, sample_seed)
            mask = mask.transpose((2, 1, 0))
            mask = torch.from_numpy(mask)
            mask = mask.unsqueeze(1)            # [frame, 1, ky, kx]
        

        # masking, transformation, normalization, and make complex dim explicit [frame, coil, ky, kx] -> [2, frame, coil, ky, kx]
        # fair normalization setting, as typically one would not have access to the fully sampled kspace
        masked_kspace = kspace * mask
        norm_factor = np.sqrt(self.undersampling_factor/2.0) # empirical normalization factor to account for undersampling
        masked_kspace = self._normalize(masked_kspace, masked_kspace.norm()*norm_factor)
        masked_kspace = torch.stack((masked_kspace.real, masked_kspace.imag), dim=0)


        name = f'{fname.parent.stem}_{dataslice}'
        return masked_kspace, mask, sensitivities, ground, name
    
class OCMRxProspective(data.Dataset):
    def __init__(
            self, 
            data_path
        ):
        super().__init__()
        self.data_path = data_path

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
            p = Path(data_path)
            if p.is_dir():
                paths = p.glob(f'**/*.h5')
            if p.is_file():
                paths = [p]
            # raise ValueError(f"{data_path} is not a valid testset.")


        # split each file to return slices individually
        self.raw_samples = []
        for fname in paths:
            with h5py.File(fname, "r") as hf:
                # num_slices = hf['kspace'].shape[1]  # count slices in [t, s, ky, kx] or [t, s, coil, ky, kx]  
                param = json.loads(hf['params'][()])
                num_slices = param['kspace_shape'][6]      

            # print(num_slices)
            if num_slices > 7: # lax has 7 slices, sax more or just one
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


    def _normalize(self, x, norm):
        x = x * np.sqrt(2*x.shape[0]*np.prod(x.shape[-2:])) / norm 
        return x

    def __len__(self):
        return len(self.raw_samples)

    def __getitem__(self, idx):
        fname, dataslice = self.raw_samples[idx]

        with h5py.File(fname, "r") as hf:
            kspace = hf['kspace'][:, dataslice]                                      # [frame, coil, ky, kx] or [frame, ky, kx]
            sensitivities = hf['sensitivities'][dataslice]                           # [coil, ky, kx]
            mask = hf['mask'][:]                                                     # [ky, kx] or [frame, slice, ky, kx]
            mask = mask[:, dataslice, None]  if mask.ndim == 4 else mask             # [frame, 1, ky, kx]
            param = json.loads(hf['params'][()])                                     # parameter information as dictionary
            
        kspace = torch.from_numpy(kspace)
        sensitivities = torch.from_numpy(sensitivities) 
        mask = torch.from_numpy(mask)    

        # masking, transformation, normalization, and make complex dim explicit [frame, coil, ky, kx] -> [2, frame, coil, ky, kx]
        # fair setting, as typically one would not have access to the fully sampled kspace
        # print(f'Acceleration factor: {1/((mask.numpy() != 0).mean())}')
        masked_kspace = kspace * mask
        norm_factor = np.sqrt((1/((mask.numpy() != 0).mean()))/2.0) #  applicable for prospectively undersampled data
        masked_kspace = self._normalize(masked_kspace, masked_kspace.norm()*norm_factor)
        masked_kspace = torch.stack((masked_kspace.real, masked_kspace.imag), dim=0)


        name = f'{fname.stem}_{dataslice}'
        return masked_kspace, mask, sensitivities, name, param
