#!/usr/bin/python

import h5py
import numpy as np
import progressbar
import os
import glob

import evaluation
import hybridtrack

E_threshold = 350

loadpattern = 'MultiAngle_HT_*.h5'
# LRC
loadpath = '/global/home/users/bcplimley/multi_angle/HTbatch01_h5'
# # Vostro
# loadpath = '/home/plimley/gh/etrack/reconstruction'

flist = glob.glob(os.path.join(loadpath, loadpattern))

pbar = progressbar.ProgressBar(
    widgets=[progressbar.Percentage(), ' ', progressbar.Bar(), ' ',
             progressbar.ETA()], maxval=len(flist))

true_alpha = np.zeros(1000000)
matlab_alpha = {'2.5': np.zeros(1000000), '10.5': np.zeros(1000000)}
n = 0

pbar.start()
for i, filename in enumerate(flist):
    h5file = h5py.File(os.path.join(loadpath, filename), 'r')

    for j in range(1000):
        eventname = '{:05d}'.format(j)
        if eventname not in h5file:
            continue
        evt = h5file[eventname]
        if 'Etot' not in evt.attrs or 'Edep' not in evt.attrs:
            continue
        if 'cheat_alpha' not in evt.attrs:
            continue
        if 'pix2_5noise0' not in evt or 'pix10_5noise0' not in evt:
            continue
        if evt.attrs['Etot'] < E_threshold or evt.attrs['Edep'] < E_threshold:
            continue
        p2 = evt['pix2_5noise0']
        p10 = evt['pix10_5noise0']
        if 'matlab_alpha' not in p2.attrs or 'matlab_alpha' not in p10.attrs:
            continue

        true_alpha[n] = evt.attrs['cheat_alpha']
        matlab_alpha['2.5'][n] = p2.attrs['matlab_alpha']
        matlab_alpha['10.5'][n] = p10.attrs['matlab_alpha']
        n += 1

    pbar.update(i)
pbar.finish()

true_alpha = true_alpha[:n]
matlab_alpha['2.5'] = matlab_alpha['2.5'][:n]
matlab_alpha['10.5'] = matlab_alpha['10.5'][:n]

np.savez(
    'lrc_h5_query_2.5.npz',
    true_alpha=true_alpha,
    matlab_alpha_2=matlab_alpha['2.5'],
    matlab_alpha_10=matlab_alpha['10.5'])
