#!/bin/python

# import tables
import h5py
import numpy as np
import progressbar
import os

import hybridtrack

# loadpath = '/mnt/data/MATLAB/data/Electron Track/geant4/'
# loadfile = 'Mat_CCD_SPEC_500k_49_TRK.h5'
loadpath = '/mnt/data/MATLAB/data/Electron Track/algorithms/results/2013sep binned'
loadfile = 'MultiAngle_HT_100_1.h5'

# widgets = [pbar.Percentage(), ' ', pbar.Bar(), ' ', pbar.ETA()]
# pb = pbar.ProgressBar(widgets=widgets, maxval=1000)
#
# pb.start()
# for i in range(1000):
#     time.sleep(0.05)
#     pb.update(i+1)
# pb.finish()

h5file = h5py.File(os.path.join(loadpath,loadfile),'r')

pbar = progressbar.ProgressBar(
    widgets=[progressbar.Percentage(), ' ', progressbar.Bar(), ' ',
    progressbar.ETA()], maxval = len(h5file))
pbar.start()

info = [[] for __ in range(len(h5file))]
py_alpha = np.zeros(len(h5file))
mat_alpha = np.zeros(len(h5file))

for i in range(len(h5file)):
    eventname = 'event{:05d}'.format(i)
    if eventname in h5file:
        event = h5file[eventname]
        trackname = 'trackimage2.5'
        # print(trackname)
        # print(event)
        if trackname in event:
            this_image_object = event[trackname]
            this_image = np.zeros(this_image_object.shape)
            this_image_object.read_direct(this_image)
            aname = 'matlab_alpha'
            if aname in event[trackname].attrs:
                mat_alpha[i] = event[trackname].attrs[aname]
            try:
                # reverse the transpose MATLAB did during h5create
                __, info[i] = hybridtrack.reconstruct(this_image.T,
                    pixel_size_um=2.5)
                py_alpha[i] = info[i].alpha_deg
            except hybridtrack.InfiniteLoop:
                print('Infinite loop on event #{}, track #{}'.format(i,j))
            except hybridtrack.NoEndsFound:
                print('No ends found on event #{}, track #{}'.format(i,j))
            except err:
                print('Other error:')
                print(err)

    pbar.update(i)
pbar.finish()

np.save('alpha2.5.npy',alpha)
