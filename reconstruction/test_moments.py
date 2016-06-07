# -*- coding: utf-8 -*-

# test_moments.py
# 2016.06.07
# for testing trackmoments.py

from __future__ import print_function
import ipdb as pdb
import numpy as np
import matplotlib.pyplot as plt
import h5py
import pandas as pd

from etrack.reconstruction.trackdata import Track
import etrack.io.trackio as trackio
import etrack.reconstruction.trackmoments as tm
from etrack.visualization.trackplot import plot_track_image


def tracklist_from_h5(filename, energy_thresh_kev):
    """
    Go through a pyml h5 file and make Tracks from pix10_5noise0noise0.
    Save them into a list if their energy is above energy_thresh_kev.
    """

    f = h5py.File(filename, 'r')
    tracklist = []
    for evtkey in f.keys():
        try:
            this_track_obj = f[evtkey][fn]
        except KeyError:
            # no pix10_5noise0
            continue
        try:
            t = Track.from_hdf5(this_track_obj)
        except trackio.InterfaceError:
            # 'HDF5 object should have an attribute, obj_type'
            # some error in read/write of hdf5?
            continue
        if t.energy_kev > energy_thresh:
            tracklist.append(t)


filename = '/media/plimley/TEAM 7B/HTbatch01_pyml/MultiAngle_HT_100_11_py.h5'
fn = 'pix10_5noise0'

energy_thresh = 300     # keV
print('Compiling tracklist (energy_thresh = {} keV)'.format(energy_thresh))
tracklist = tracklist_from_h5(filename, energy_thresh)

print('Testing moments on entire tracks')
index_error_count = 0
for t in tracklist:
    try:
        tm.MomentsReconstruction.reconstruct_test(t.image, 0)
    except IndexError:
        index_error_count += 1

print('{} IndexErrors out of {} tracks'.format(
    index_error_count, len(tracklist)))
