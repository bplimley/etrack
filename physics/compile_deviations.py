#!/bin/python
# compile_deviations.py
# runs after devScript.py

# before: every Mat_* file has a dev_*.npz file with range and angles
# after: make a file for every 100 input files, organized by energy

# energy_keV is rounded to an INTEGER. because I have weird float values.

import numpy as np
import glob
import os
import sys
import time
import datetime

def get_energy_list():
    # calculating energy from geant4 is unreliable.
    # so use this list instead.
    energy_list1 = [20,30,40,50,60,70,80,90,100,
                    120,140,160,180,200,250,300,350,400,450,500,
                    600,700,800,900,1000,
                    1200,1400,1600,1800,2000]
    energy_list2 = [];
    for E in energy_list1:
        for i in xrange(10):
            energy_list2.append(E)
    return energy_list2



batch_size = 100
elements_per_file = 300
warn_switch = True

skip_flag = False
# True allows it to skip files that were already completed

baseDir = '/global/scratch/bcplimley/electrons/parameterTests/'
dirform = 'e_Si*'
dirlist = glob.glob(os.path.join(baseDir,dirform))
# dirlist = [os.path.split(d)[-1] for d in dirlist]

for d in dirlist:
    # dirfull = os.path.join(...)

    print 'Entering directory ' + d + ' at ' + time.ctime()

    filepattern = 'dev_*.npz'
    flist = glob.glob(os.path.join(d,filepattern))
    fname = [os.path.basename(f) for f in flist]

    # build batches
    n_batches = int(np.ceil(len(flist) / np.float(batch_size)))
    batch_start = range(0,len(flist),batch_size)
    batch_stop = range(batch_size,len(flist),batch_size)
    batch_stop.append(len(flist))

    print 'Set up ' + str(n_batches) + ' batches in ' + d + ' at ' + time.ctime()

    for b in xrange(n_batches):
        batchname = '_'.join(('devbatch',str(b).zfill(4),
                              fname[batch_start[b]][10:]))
        batchfull = os.path.join(d,batchname)
        if os.path.isfile(batchfull) and skip_flag:
            # already has *.npz suffix
            continue

        datasize = xrange(batch_size*elements_per_file)
        b_energy_keV = [0 for i in datasize]
        b_distance_mm = [[] for i in datasize]
        b_deviation_deg = [[] for i in datasize]

        ind = 0
        for f in flist[batch_start[b]:batch_stop[b]]:
            data = np.load(f)
            # data['energy_keV']
            # data['radial_distance_mm']
            # data['deviation_deg']
            if warn_switch and len(data['energy_keV']
                                  ) != elements_per_file:
                raise RuntimeError(' '.join((
                    'Found',str(len(data['energy_keV'])),
                    'elements per file, expected',str(elements_per_file)
                        )))
            ind2 = ind + elements_per_file
            # b_energy_keV[ind:ind2] = data['energy_keV'].astype(int)
            b_energy_keV[ind:ind2] = get_energy_list()
            b_distance_mm[ind:ind2] = data['radial_distance_mm']
            b_deviation_deg[ind:ind2] = data['deviation_deg']
            ind = ind2

        np.savez(batchfull,
                 b_energy_keV=b_energy_keV,
                 b_distance_mm=b_distance_mm,
                 b_deviation_deg=b_deviation_deg)

        # Write the same data to disk, but organized by energy.
        # This enables each energy to later be compiled separately,
        #   with less concern for running out of memory.

        unique_energies = np.unique(b_energy_keV)
        if len(unique_energies) != 30:
            raise RuntimeError(' '.join((
                'Found',str(len(unique_energies)),
                'energies instead of 30!')))
        bE_distance_mm = [[] for u in unique_energies]
        bE_deviation_deg = [[] for u in unique_energies]

        for u in unique_energies:
            list_of_lengths = [len(b_distance_mm[i])
                               for i in xrange(len(b_distance_mm))
                               if b_energy_keV[i]==u]
            list_of_distance = [b_distance_mm[i]
                                for i in xrange(len(b_distance_mm))
                                if b_energy_keV[i]==u]
            list_of_deviation = [b_deviation_deg[i]
                                 for i in xrange(len(b_distance_mm))
                                 if b_energy_keV[i]==u]
            total_length = sum(list_of_lengths)
            bE_distance_mm = np.zeros(total_length)
            bE_deviation_deg = np.zeros(total_length)
            ind3 = 0
            for i in xrange(list_of_lengths):
                ind4 = ind3 + list_of_lengths[i]
                bE_distance_mm[ind3:ind4] = list_of_distance[i]
                bE_deviation_deg[ind3:ind4] = list_of_deviation[i]
            bE_name = '_'.join(('devbatch',str(b).zfill(4),
                                'E',str(u).zfill(4),
                                fname[batch_start[b]][10:]))
            bE_full = os.path.join(d,bE_name)
            np.savez(bE_full,
                     energy_of_this_file = u,
                     list_of_lengths = list_of_lengths,
                     bE_distance_mm = bE_distance_mm,
                     bE_deviation_deg = bE_deviation_deg)


        print 'Done with ' + batchname + ' at ' + time.ctime()


