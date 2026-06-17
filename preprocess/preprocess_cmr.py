import argparse
import torch
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr

from bart import bart
from pathlib import Path

from cardiac_diffusion.mri import kspace2image # computes the MVUE image

def process_all(data_path, output_path, inspect=False):
    # process all files in the dataset
    
    files = data_path.glob(f'**/*.mat')
    
    for file_path in files:
        try:
            process_file(file_path, output_path, inspect=inspect)
        except:
            print(f'Error processing {file_path}. Skipping file.', flush=True)
        finally:
            continue

def process_file(file_path, output_path, inspect=False):
    # process one file
    # load kspace -> get sensitivity -> get MVUE (-> if inspect: plot everything) -> save to h5

    # load kspace
    kspace = load_mat(file_path, key='kspace_full')
    kspace = kspace['real'] + 1j * kspace['imag']

    # compute sensitivity maps
    sensitivities = sliced_espirit(kspace, m=1, c=0)

    # compute MVUE images
    mvue = kspace2image(torch.from_numpy(kspace), sens_maps=torch.from_numpy(sensitivities), fdim=(3, 4), cdim=2).numpy()

    # combine data
    data = (kspace, sensitivities, mvue)
    if inspect:
        inspect_data(data)

    # save to h5
    subject = output_path / file_path.parent.stem
    subject.mkdir(exist_ok=True)
    file_output = subject / f'{file_path.stem}.h5'
    write_h5(file_output, data)

def load_mat(fname, key=None):
    with h5py.File(fname, "r") as hf:
        key = list(hf.keys())[0] if key is None else key
        data = np.asarray(hf[key])

    return data

def write_h5(fname, data):
    kspace, sensitivities, mvue = data
    
    with h5py.File(fname, 'w') as hf:
        hf.create_dataset('kspace', data=kspace)
        hf.create_dataset('sensitivities', data=sensitivities)
        hf.create_dataset('image', data=mvue)

    print(f'Created Dataset {fname.parent.stem}/{fname.stem}.', flush=True)

def inspect_data(data):
    # print infos and plot data (kspace, sensitivities, mvue)
    kspace, sensitivities, mvue = data
    
    print('kspace:')
    print(f'shape: {kspace.shape}, [frame, slice, coil, ky, kx]')
    print(f'dtype: {kspace.dtype}, complex64')
    fidx = kspace.shape[0] // 2
    sidx = kspace.shape[1] // 2

    fig, _ = plt.subplots(5, 2) 
    for i, ax in enumerate(fig.axes):
        ax.imshow(abs(kspace[fidx, sidx, i]), cmap='gray', norm=clr.PowerNorm(gamma=0.25))
        ax.axis('off')
    plt.show()
    plt.close(fig)
    
    print('sensitivity maps:')
    print(f'shape: {sensitivities.shape}, [slice, coil, ky, kx]')
    print(f'dtype: {sensitivities.dtype}, complex64')

    fig, _ = plt.subplots(5, 2)
    for i, ax in enumerate(fig.axes):
        ax.imshow(abs(sensitivities[sidx, i]), cmap='gray')
        ax.axis('off')
    plt.show()
    plt.close(fig)
    
    print('MVUE images:')
    print(f'shape: {mvue.shape}, [frame, slice, ky, kx]')
    print(f'dtype: {mvue.dtype}, complex64')

    fig, _ = plt.subplots(6, 2)
    for i, ax in enumerate(fig.axes):
        try:
            ax.imshow(abs(mvue[i, sidx]), cmap='gray')
            ax.axis('off')
        except IndexError:
            break
    plt.show()
    plt.close(fig)

    plt.imshow(abs(mvue[fidx, sidx]), cmap='gray')
    plt.axis('off')
    plt.show()

    print('RSS image as reference:')
    rssi = kspace2image(torch.from_numpy(kspace), fdim=(3, 4), cdim=2, adaptive=False).numpy()
    plt.imshow(abs(rssi[fidx, sidx]), cmap='gray')
    plt.axis('off')
    plt.show()


def sliced_espirit(kspace: np.ndarray,  **kwargs):
    '''ESPIRiT sensitivity map estimation using BART ecalib.

    Inputs:
        - kspace            : accelerated kspace
        - kwargs            : for the usage of flags e.g. for flag -a give a='', for -m 3 give m=3, ...

    Output:
        - sensitivities     : sensitivity maps belonging to the given kspace

    _____________________________________________________________________________________________________________________________________________________

    Usage: ecalib [-t f] [-c f] [-k d:d:d] [-r d:d:d] [-m d] [-S] [-W] [-I] [-1] [-P] [-v f] [-a] [-d d] <kspace> <sensitivities> [<ev-maps>]

    Estimate coil sensitivities using ESPIRiT calibration.
    Optionally outputs the eigenvalue maps.

    -t threshold     This determined the size of the null-space. 
    -c crop_value    Crop the sensitivities if the eigenvalue is smaller than {crop_value}.
    -k ksize         kernel size
    -r cal_size      Limits the size of the calibration region.
    -m maps          Number of maps to compute.
    -S               create maps with smooth transitions (Soft-SENSE).
    -W               soft-weighting of the singular vectors.
    -I               intensity correction
    -1               perform only first part of the calibration
    -P               Do not rotate the phase with respect to the first principal component
    -v variance      Variance of noise in data.
    -a               Automatically pick thresholds.
    -d level         Debug level
    -h               help

    _____________________________________________________________________________________________________________________________________________________
    '''
    # transform input to BART
    # kspace: [frame, slice, coil, ky, kx] -> [kx, ky, coil, frame, slice] -> [kx, ky, kz=1, coil, map=1, ?, ?, ?, ?, ?, frame, ?, ?, slice]
    kspace_bart = kspace.transpose(4, 3, 2, 0, 1)
    # kspace_bart = kspace_bart[:, :, None, :, None, None, None, None, None, None, :, None, None, :]
    # BART does not return different sensitivities for different slices -> especially bad for lax

    # do BART sensitivity estimation
    command = 'ecalib '
    for kw in kwargs:
        command += f'-{kw} {kwargs[kw]} '

    sliced_sensitivities = []
    for s in range(kspace_bart.shape[-1]):
        slice_kspace = kspace_bart[:, :, None, :, None, None, None, None, None, None, :, None, None, s]
        sensitvities = bart(1, command, slice_kspace)
        sliced_sensitivities.append(sensitvities.transpose(3, 2, 1, 0).squeeze(1))

    # transform to output shape
    sensitvities = np.stack(sliced_sensitivities, axis=0)

    return sensitvities


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--base", type=Path, required=True, metavar="/path/to/base_directory", help='Path to the base_directory.')
    parser.add_argument("-d", '--data', type=Path, default=None, metavar="/path/to/data_directory", 
                        help='Path to the dataset to preprocess. [Optional] Defaults to: /path/to/base_directory/datasets.')
    args = parser.parse_args()

    BASE = args.base
    if args.data is None:
        DATA = BASE / 'datasets/'
    elif not args.data.is_dir():
        print(f'Provided data path {args.data} is not a directory. Exiting.', flush=True)
        exit(1)
    OUTPUT = BASE / 'datasets/CineProcessed/'


    ### 2023 data

    data = DATA / 'CMRxRecon2023/ChallengeData/MultiCoil/Cine/TrainingSet/FullSample'
    if data.is_dir():
        output = OUTPUT / '2023/TrainingSet'
        output.mkdir(parents=True, exist_ok=True)
        print(f'Processing {data} to {output}...')
        process_all(data, output)
    else:
        print(f'Data path {data} is not a directory. Skipping.')

    data = DATA / 'CMRxRecon2023/ChallengeDataAfterCompetition/ChallengeData_validation/MultiCoil/Cine/ValidationSet/FullSample'
    if data.is_dir():
        output = OUTPUT / '2023/ValidationSet'
        output.mkdir(parents=True, exist_ok=True)
        print(f'Processing {data} to {output}...')
        process_all(data, output)
    else:
        print(f'Data path {data} is not a directory. Skipping.')
    
    data = DATA / 'CMRxRecon2023/ChallengeDataAfterCompetition/ChallengeData_test/MultiCoil/Cine/TestSet/FullSample'
    if data.is_dir():
        output = OUTPUT / '2023/TestSet'
        output.mkdir(parents=True, exist_ok=True)
        print(f'Processing {data} to {output}...')
        process_all(data, output)
    else:
        print(f'Data path {data} is not a directory. Skipping.')


    ### 2024 data

    data = DATA / 'CMRxRecon2024/ChallengeData/MultiCoil/Cine/TrainingSet/FullSample'
    if data.is_dir():
        output = OUTPUT / '2024/TrainingSet'
        output.mkdir(parents=True, exist_ok=True)
        print(f'Processing {data} to {output}...')
        process_all(data, output)
    else:
        print(f'Data path {data} is not a directory. Skipping.')

    data = DATA / 'CMRxRecon2024/ChallengeDataAfterCompetition/MultiCoil/Cine/ValidationSet/FullSample'
    if data.is_dir():
        output = OUTPUT / '2024/ValidationSet'
        output.mkdir(parents=True, exist_ok=True)
        print(f'Processing {data} to {output}...')
        process_all(data, output)
    else:
        print(f'Data path {data} is not a directory. Skipping.')
    
    data = DATA / 'CMRxRecon2024/ChallengeDataAfterCompetition/MultiCoil/Cine/TestSet/FullSample'
    if data.is_dir():
        output = OUTPUT / '2024/TestSet'
        output.mkdir(parents=True, exist_ok=True)
        print(f'Processing {data} to {output}...')
        process_all(data, output)
    else:
        print(f'Data path {data} is not a directory. Skipping.')

