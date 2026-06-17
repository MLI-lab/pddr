import argparse
import torch
import h5py
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr

import os
import ismrmrd
import ismrmrd.xsd

from bart import bart
from pathlib import Path

from cardiac_diffusion.mri import kspace2image # computes the MVUE image

def process_all(data_path, output_path, undersampled_data=False, inspect=False):
    # process all files in the dataset
    
    files = data_path.glob(f'**/*.h5')
    
    for file_path in files:
        process_file(file_path, output_path, undersampled_data=undersampled_data, inspect=inspect)

def process_file(file_path, output_path, undersampled_data=False, inspect=False):
    # process one file
    # load kspace -> get sensitivity -> get MVUE -> get masks (-> if inspect: plot everything) -> save to h5

    # load kspace
    kspace, param = read_ocmr(file_path)

    param['IRes'] = [param['FOV'][0]/kspace.shape[0], param['FOV'][1]/kspace.shape[1], param['FOV'][2]/kspace.shape[2]]
    param['kspace_dim'] = list(param['kspace_dim'])
    param['kspace_shape'] = list(kspace.shape)

    try:
        assert kspace.shape[2] == 1, 'kz is not 1.'
        assert kspace.shape[5] == 1, 'set is not 1.'
        assert kspace.shape[7] == 1, 'rep is not 1.'
        assert kspace.shape[8] == 1, 'avg is not 1.'
    except AssertionError:
        return
    
    one_slice = kspace.shape[6] == 1
    kspace = kspace.squeeze()
    kspace = kspace[..., None] if one_slice else kspace
    kspace = kspace.transpose(3, 4, 2, 1, 0) # [kx, ky, coil, frame, slice] -> [frame, slice, coil, ky, kx]
    time_avg = np.mean(kspace, axis=0, keepdims=True)

    # compute sensitivity maps on time average
    sensitivities = sliced_espirit(time_avg, m=1, c=0)

    # compute MVUE images
    if not undersampled_data:
        mvue = kspace2image(torch.from_numpy(kspace), sens_maps=torch.from_numpy(sensitivities), fdim=(3, 4), cdim=2).numpy()

    # get mask from undersampled kspace
    if undersampled_data:
        mask = (abs(np.mean(kspace, axis = 2)) > 0).astype(np.float32)

    # combine data
    if undersampled_data:
        data = (kspace, sensitivities, mask, param)
    else:
        data = (kspace, sensitivities, mvue, param)

    if inspect:
        inspect_data(data, undersampled_data=undersampled_data)

    # save to h5
    file_output = output_path / f'{file_path.stem}.h5'
    write_h5(file_output, data, undersampled_data=undersampled_data)

def write_h5(fname, data, undersampled_data=False):

    if undersampled_data:
        kspace, sensitivities, mask, param = data

        with h5py.File(fname, 'w') as hf:
            hf.create_dataset('kspace', data=kspace)
            hf.create_dataset('sensitivities', data=sensitivities)
            hf.create_dataset('mask', data=mask)
            hf.create_dataset('params', data=json.dumps(param).encode('utf-8'))

    else:
        kspace, sensitivities, mvue, param = data
        
        with h5py.File(fname, 'w') as hf:
            hf.create_dataset('kspace', data=kspace)
            hf.create_dataset('sensitivities', data=sensitivities)
            hf.create_dataset('image', data=mvue)
            hf.create_dataset('params', data=json.dumps(param).encode('utf-8'))

    print(f'Created Dataset {fname.parent.stem}/{fname.stem}.', flush=True)

def inspect_data(data, undersampled_data=False):
    # print infos and plot data (kspace, sensitivities, mvue, mask)
    if undersampled_data:
        kspace, sensitivities, mask, param = data
        mvue = None
    else:
        kspace, sensitivities, mvue, param = data

    print('Parameters:')
    print(param)
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

    if mvue is not None:
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

    if undersampled_data:
        print('mask:')
        print(f'shape: {mask.shape}, [frame, slice, ky, kx]')
        print(f'dtype: {mask.dtype}, float32')
        print(f'actual factor: {1/((mask != 0).mean())}')
    
        fig1 = plt.figure(1); fig1.suptitle("Sampling Pattern", fontsize=14)
        plt.subplot2grid((1, 8), (0, 0), colspan=6)
        tmp = plt.imshow(mask[fidx, sidx], cmap='gray', aspect= 'auto')
        plt.xlabel('kx');plt.ylabel('ky'); tmp.set_clim(0.0,1.0) # ky by kx
        plt.subplot2grid((1, 9), (0, 7),colspan=2)
        tmp = plt.imshow(mask[:, sidx, :, mask.shape[3]//2].T, cmap='gray', aspect= 'auto')
        plt.xlabel('frame');plt.yticks([]); tmp.set_clim(0.0, 1.0) # ky by frame
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

def read_ocmr(filename):
# Before running the code, install ismrmrd-python and ismrmrd-python-tools:
#  https://github.com/ismrmrd/ismrmrd-python
#  https://github.com/ismrmrd/ismrmrd-python-tools
#
# Input:  *.h5 file name
# Output: all_data    k-space data, orgnazide as {'kx'  'ky'  'kz'  'coil'  'phase'  'set'  'slice'  'rep'  'avg'}
#         param  some parameters of the scan
# 

# This is a function to read K-space from ISMRMD *.h5 data
# from https://github.com/ismrmrd/ismrmrd-python-tools/blob/master/recon_ismrmrd_dataset.py

    if not os.path.isfile(filename):
        print("%s is not a valid file" % filename)
        raise SystemExit
    dset = ismrmrd.Dataset(filename, 'dataset', create_if_needed=False)
    header = ismrmrd.xsd.CreateFromDocument(dset.read_xml_header())
    enc = header.encoding[0]

    # Matrix size
    eNx = enc.encodedSpace.matrixSize.x
    #eNy = enc.encodedSpace.matrixSize.y
    eNz = enc.encodedSpace.matrixSize.z
    eNy = (enc.encodingLimits.kspace_encoding_step_1.maximum + 1); #no zero padding along Ny direction

    # Field of View
    eFOVx = enc.encodedSpace.fieldOfView_mm.x
    eFOVy = enc.encodedSpace.fieldOfView_mm.y
    eFOVz = enc.encodedSpace.fieldOfView_mm.z
    
    # Save the parameters    
    param = dict();
    param['TRes'] =  str(header.sequenceParameters.TR)
    param['FOV'] = [eFOVx, eFOVy, eFOVz]
    param['TE'] = str(header.sequenceParameters.TE)
    param['TI'] = str(header.sequenceParameters.TI)
    param['echo_spacing'] = str(header.sequenceParameters.echo_spacing)
    param['flipAngle_deg'] = str(header.sequenceParameters.flipAngle_deg)
    param['sequence_type'] = header.sequenceParameters.sequence_type

    # Read number of Slices, Reps, Contrasts, etc.
    nCoils = header.acquisitionSystemInformation.receiverChannels
    try:
        nSlices = enc.encodingLimits.slice.maximum + 1
    except:
        nSlices = 1
        
    try:
        nReps = enc.encodingLimits.repetition.maximum + 1
    except:
        nReps = 1
               
    try:
        nPhases = enc.encodingLimits.phase.maximum + 1
    except:
        nPhases = 1;

    try:
        nSets = enc.encodingLimits.set.maximum + 1;
    except:
        nSets = 1;

    try:
        nAverage = enc.encodingLimits.average.maximum + 1;
    except:
        nAverage = 1;   
        
    # TODO loop through the acquisitions looking for noise scans
    firstacq=0
    for acqnum in range(dset.number_of_acquisitions()):
        acq = dset.read_acquisition(acqnum)

        # TODO: Currently ignoring noise scans
        if acq.isFlagSet(ismrmrd.ACQ_IS_NOISE_MEASUREMENT):
            #print("Found noise scan at acq ", acqnum)
            continue
        else:
            firstacq = acqnum
            print("Imaging acquisition starts acq ", acqnum)
            break

    # assymetry echo
    kx_prezp = 0;
    acq_first = dset.read_acquisition(firstacq)
    if  acq_first.center_sample*2 <  eNx:
        kx_prezp = eNx - acq_first.number_of_samples
         
    # Initialiaze a storage array
    param['kspace_dim'] = {'kx ky kz coil phase set slice rep avg'};
    all_data = np.zeros((eNx, eNy, eNz, nCoils, nPhases, nSets, nSlices, nReps, nAverage), dtype=np.complex64)
    
    # check if pilot tone (PT) is on
    pilottone = 0;
    try:
        if (header.userParameters.userParameterLong[3].name == 'PilotTone'):
            pilottone = header.userParameters.userParameterLong[3].value;
    except:
        pilottone = 0;  
            
    if pilottone == 1:
        print('Pilot Tone is on, discarding the first 3 and last 1 k-space point for each line')

    # Loop through the rest of the acquisitions and stuff
    for acqnum in range(firstacq,dset.number_of_acquisitions()):
        acq = dset.read_acquisition(acqnum)
        if pilottone == 1: # discard the first 3 and last 1 k-space point to exclude PT artifact
            acq.data[:,[0,1,2,acq.data.shape[1]-1]] = 0        

        # Stuff into the buffer
        y = acq.idx.kspace_encode_step_1
        z = acq.idx.kspace_encode_step_2
        phase =  acq.idx.phase;
        set =  acq.idx.set;
        slice =  acq.idx.slice;
        rep =  acq.idx.repetition;
        avg = acq.idx.average;        
        all_data[kx_prezp:, y, z, :,phase, set, slice, rep, avg ] = np.transpose(acq.data)
        
    return all_data,param

def download(download_path='/mnt/hdd_pool_bigsur/datasets/OCMR/ocmr/', **kwargs):
    """
    Download the OCMR data to the specified download_path.
    The data is filtered based on the input arguments, e.g. dur='lng' filters for long sequences, smp='fs' for fullysampled data...
    """
    import pandas
    # !python -m pip install --disable-pip-version-check -qq --upgrade boto3
    # #!python -m pip list --disable-pip-version-check | grep -w 'boto3 '
    import boto3
    from botocore import UNSIGNED
    from botocore.client import Config

    # filter data
    ocmr_data_attributes_location = '/mnt/hdd_pool_bigsur/datasets/OCMR/ocmr/ocmr_data_attributes.csv'
    df = pandas.read_csv(ocmr_data_attributes_location)
    # Cleanup empty rows and columns
    df.dropna(how='all', axis=0, inplace=True)
    df.dropna(how='all', axis=1, inplace=True)
    # healthy_df = df.query ('dur=="lng" and fov=="noa" and sub=="vol"', engine='python')
    # patient_df = df.query ('dur=="lng" and fov=="noa" and sub=="pat"', engine='python')

    # filter the data based on the input arguments
    command = ''
    for kw in kwargs:
        command += f'{kw}=="{kwargs[kw]}" and '
    command = command[:-5]
    print(command)
    selected_df = df.query(command, engine='python')
    print(len(selected_df))
    print(selected_df)

    # download data
    # Replace this with the name of the OCMR S3 bucket 
    bucket_name = 'ocmr'
        
    count=1
    s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))

    # Iterate through each row in the filtered DataFrame and download the file from S3. 
    # Note: Test after finalizing data in S3 bucket
    for index, row in selected_df.iterrows():
        print('Downloading {} to {} (File {} of {})'.format(row['file name'], download_path, count, len(selected_df)))
        s3_client.download_file(bucket_name, 'data/{}'.format(row['file name']), '{}/{}'.format(download_path,row['file name']))
        count+=1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--base", type=Path, required=True, metavar="/path/to/base_directory", help='Path to the base_directory.')
    parser.add_argument("-d", '--data', type=Path, default=None, metavar="/path/to/data_directory", 
                        help='Path to the store the dataset. [Optional] Defaults to: /path/to/base_directory/datasets.')
    args = parser.parse_args()

    BASE = args.base
    if args.data is None:
        DATA = BASE / 'datasets/'
    elif not args.data.is_dir():
        print(f'Provided data path {args.data} is not a directory. Exiting.', flush=True)
        exit(1)
    OUTPUT = BASE / 'datasets/CineProcessed/'


    ### Prospective long data of healthy volunteers

    data = DATA / 'OCMR/prospective/healthy'
    # data.mkdir(parents=True, exist_ok=True)
    # download(download_path=data, dur='lng', fov='noa', sub='vol') 

    output = OUTPUT / 'OCMR/prospective/healthy'
    output.mkdir(parents=True, exist_ok=True)
    print(f'Processing {data} to {output}...')
    process_all(data, output, undersampled_data=True)


    ### Prospective long data of patients

    data = DATA / 'OCMR/prospective/patient'
    # data.mkdir(parents=True, exist_ok=True)
    # download(download_path=data, dur='lng', fov='noa', sub='pat') 

    output = OUTPUT / 'OCMR/prospective/patient'
    output.mkdir(parents=True, exist_ok=True)
    print(f'Processing {data} to {output}...')
    process_all(data, output, undersampled_data=True)


    # ### Prospective short data of healthy volunteers

    # data = DATA / 'OCMR/prospective/short'
    # data.mkdir(parents=True, exist_ok=True)
    # download(download_path=data, smp='pse', dur='shr', fov='noa', sub='vol') 

    # output = OUTPUT / 'OCMR/prospective/short'
    # output.mkdir(parents=True, exist_ok=True)
    # print(f'Processing {data} to {output}...')
    # process_all(data, output, undersampled_data=True)


    # ### Fullysampled (binned) data

    # data = DATA / 'OCMR/fullysampled'
    # data.mkdir(parents=True, exist_ok=True)
    # download(download_path=data, smp='fs', fov='noa') 

    # output = OUTPUT / 'OCMR/fullysampled'
    # output.mkdir(parents=True, exist_ok=True)
    # print(f'Processing {data} to {output}...')
    # process_all(data, output, undersampled_data=False)
