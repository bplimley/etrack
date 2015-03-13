#!/bin/python
# compile_deviations.py
# runs after devScript.py

# before: every Mat_* file has a dev_*.npz file with range and angles
# after: make a file for every 100 input files, organized by energy

import numpy as np
import glob
import os
import sys
import time
import datetime


batch_size = 100
elements_per_file = 300
warn_switch = True

baseDir = '/global/scratch/bcplimley/electrons/parameterTests/'
dirform = 'e_Si*'
dirlist = glob.glob(os.path.join(baseDir,dirform))
# dirlist = [os.path.split(d)[-1] for d in dirlist]

for d in dirlist:
    # dirfull = os.path.join(...)

    print 'Entering directory ' + d + ' at ' + time.ctime()

    filepattern = 'dev_*.npz'
    flist = glob.glob(os.path.join(d,filepattern))

    # build batches
    n_batches = int(np.ceil(len(flist) / np.float(batch_size)))
    batch_start = range(0,len(flist),batch_size)
    batch_stop = range(batch_size,len(flist),batch_size)
    batch_stop.append(len(flist))

    print 'Set up ' + str(n_batches) + ' batches in ' + d + ' at ' + time.ctime()

    for b in xrange(n_batches):
        batchname = 'devbatch_' + str(b).zfill(4) + flist[batch_start[b]][10:]
        batchfull = os.path.join(d,batchname)
        if os.path.isfile(batchfull): # already has *.npz suffix
            continue

        b_energy_keV = np.zeros(batch_size*elements_per_file)
        b_distance_mm = np.zeros(batch_size*elements_per_file)
        b_deviation_deg = np.zeros(batch_size*elements_per_file)
        ind = 0

        for f in flist[batch_start[b]:batch_stop[b]]:
            with open(f) as data:
                # data['energy_keV']
                # data['radial_distance_mm']
                # data['deviation_deg']
                if warn_switch and len(data['energy_keV']
                                      ) is not elements_per_file:
                    raise RuntimeError(
                        'Incorrect number of elements per file!')
                ind2 = ind + elements_per_file
                b_energy_keV[ind:ind2] = data['energy_keV']
                b_distance_mm[ind:ind2] = data['radial_distance_mm']
                b_deviation_deg[ind:ind2] = data['deviation_deg']
                ind = ind2

        np.savez(batchfull,
                 b_energy_keV=b_energy_keV,
                 b_distance_mm=b_distance_mm,
                 b_deviation_deg=b_deviation_deg)
        print 'Done with ' + batchname + ' at ' + time.ctime()
