#!/usr/bin/env python3
'''
    Charge/light association file:
    
    Takes an hdf5 light event file and an hdf5 charge event file (evd) and
    generates a list of associations between the external triggers in 
    the charge event file and the events in the light file
    
    Also stores a soft-link to each dataset within the two files (a fast
    copy of the datasets is on the horizon).
    
    Newly added data structure:
    
    event_assoc - attributes:
                    - assoc_dset_ref : (2,) reference to each dataset that was linked
                - shape [i8] (N, 2) 0 index into assoc_dset[0], 0 index into assoc_dset[1]
                
    Minimal example for accessing associated data from different datasets:
        assoc = f['event_assoc']
        dset0_ref = f['event_assoc'].attrs['assoc_dset_ref'][0]
        dset1_ref = f['event_assoc'].attrs['assoc_dset_ref'][1]
        dset0 = f[dset0_ref][:] # shape (N, ...)
        dset1 = f[dset1_ref][:] # shape (M, ...)
        
        assoc_dset0 = dset0[assoc[:,0]] # shape (L, ...)
        assoc_dset1 = dset1[assoc[:,1]] # shape (L, ...)
        
    Script usage:
        ./charge_light_association.py -l <light h5 event file> -c <charge h5 event file> -o <output h5 file> \
            --ts_window <max delta t in ticks to consider associated, opt.> \
            [--copy] # flag to copy data rather than just soft-links
        
'''
import h5py
import numpy as np
import os
import argparse
from argparse import RawDescriptionHelpFormatter
from tqdm import tqdm
import subprocess

_default_ts_window = 1000

_block_size = 256

def copy_file(source, dest):
    keys = []
    with h5py.File(source, 'r') as lf:
        keys = list(lf.keys())
    for key in tqdm(keys, desc='Copy {}'.format(source)):
        rv = subprocess.run([
            'h5copy','-f','ref','--enable-error-stack',
            '-i',source,'-o',dest,
            '-s',key,'-d',key
        ])
        rv.check_returncode()

def main(light_event_filename, charge_event_filename, output_filename, copy=False, ts_window=_default_ts_window, **kwargs):
    if os.path.exists(output_filename):
        raise RuntimeError(f'{output_filename} exists! Refusing to overwrite.')
        
    # write links to datasets
    if not copy:
        with h5py.File(light_event_filename, 'r') as lf:
            with h5py.File(charge_event_filename, 'r') as cf:
                    with h5py.File(output_filename, 'w') as of:
                        for key in lf:
                            of[key] = h5py.ExternalLink(light_event_filename, key)
                        for key in cf:
                            of[key] = h5py.ExternalLink(charge_event_filename, key)
    else:
#         raise RuntimeError('Copy doesn\'t work yet, sorry')
        with h5py.File(output_filename, 'w') as of: pass
        copy_file(light_event_filename, output_filename)
        copy_file(charge_event_filename, output_filename)

    with h5py.File(output_filename, 'a') as of:  
        # perform file alignment
        charge_events = of['ext_trigs'] # charge system external triggers
        light_events = of['light_event'] # light system events

        ## unix timestamps
        print('fetching timestamps... (this is a little slow)')
        light_unix_ts = (np.max(light_events['utime_ms'],axis=-1)/1000).astype(int)
        charge_unix_ts = np.array([of['events'][ trig['event_ref'] ]['unix_ts'] for trig in tqdm(charge_events)]).astype(int)

        ## PPS timestamps
        light_ts = (np.max(light_events['tai_ns'],axis=-1)/100).astype(int)
        charge_ts = charge_events['ts'].astype(int)

        print('light data:', light_unix_ts.shape, light_ts.shape)
        print('charge data:', charge_unix_ts.shape, charge_ts.shape)

        light_n_events = light_unix_ts.shape[0]
        charge_n_events = charge_unix_ts.shape[0]
        print('events: {} (light), {} (charge)'.format(light_n_events, charge_n_events))

        light_start_time = np.min(light_unix_ts)
        charge_start_time = np.min(charge_unix_ts)
        light_end_time = np.max(light_unix_ts)
        charge_end_time = np.max(charge_unix_ts)
        print('start times: {} (light), {} (charge)'.format(light_start_time, charge_start_time))
        print('end times: {} (light), {} (charge)'.format(light_end_time, charge_end_time))

        ## first index of light file that can be associated with the charge file
        light_start_idx = np.argmax((light_unix_ts < charge_end_time) & (light_unix_ts > charge_start_time - 1))
        ## first index of charge file that can be associated with the light file
        charge_start_idx = np.argmax((charge_unix_ts < light_end_time) & (charge_unix_ts > light_start_time - 1))
        ## last index of light file that can be associated with the charge file
        light_end_idx = light_n_events - 1 - np.argmax(((light_unix_ts > charge_start_time) & (light_unix_ts < charge_end_time + 1))[::-1])
        ## last index of charge file that can be associated with the light file
        charge_end_idx = charge_n_events - 1 - np.argmax(((charge_unix_ts > light_start_time) & (charge_unix_ts < light_end_time + 1))[::-1])
        print('first index: {} (light), {} (charge)'.format(light_start_idx, charge_start_idx))
        print('last index: {} (light), {} (charge)'.format(light_end_idx, charge_end_idx))

        # and now associate
        assoc = []
        for light_curr_idx in tqdm(range(light_start_idx, light_end_idx, _block_size), desc='Matching'):
            i_light = light_curr_idx
            j_light = min(i_light+_block_size, light_end_idx)

            # first possible matching index
            i_charge = np.argmax((charge_unix_ts < light_unix_ts[j_light]) & (charge_unix_ts > light_unix_ts[i_light] - 1))
            # last possible matching index
            j_charge = charge_n_events - 1 - np.argmax(((charge_unix_ts > light_unix_ts[i_light]) & (charge_unix_ts < light_unix_ts[j_light] + 1))[::-1]) 

            assoc_mat = \
                (np.abs(light_unix_ts[i_light:j_light].reshape(1,-1) - charge_unix_ts[i_charge:j_charge].reshape(-1,1)) <= 1) \
                & (np.abs(light_ts[i_light:j_light].reshape(1,-1) - charge_ts[i_charge:j_charge].reshape(-1,1)) < ts_window)
    
            if np.any(assoc_mat):
                idcs = np.argwhere(assoc_mat) + np.array([[i_charge, i_light]])
                assoc.append(idcs)
        
        # combine arrays
        if len(assoc):
            assoc = np.concatenate(assoc, axis=0)
            assoc = np.unique(assoc.astype(int), axis=0)
        
            # create output dataset
            of.create_dataset('event_assoc', data=assoc, compression='gzip')
            of['event_assoc'].attrs['assoc_dset_ref'] = np.array([of['ext_trigs'].ref, of['light_event'].ref], dtype=h5py.ref_dtype)
        
        print('matched {} events'.format(len(assoc)))
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument('--light_event_filename','-l', type=str, required=True)
    parser.add_argument('--charge_event_filename','-c', type=str, required=True)
    parser.add_argument('--output_filename','-o', type=str, required=True)
    parser.add_argument('--copy', action='store_true', help='''Copy data from source files rather than just creating soft-links''')
    parser.add_argument('--ts_window', type=int, default=_default_ts_window, help='''Time window for association in larpix ticks [0.1us] (default=%(default)s)''')
    args = parser.parse_args()
    main(**vars(args))
