# -*- coding: utf-8 -*-

# test_moments.py
# 2016.06.07
# for testing trackmoments.py

import ipdb as pdb
import numpy as np
import matplotlib.pyplot as plt
import h5py
import pandas as pd

from etrack.reconstruction.trackdata import Track
import etrack.io.trackio as trackio
import etrack.reconstruction.trackmoments as tm
from etrack.visualization.trackplot import plot_track_image

filename = '/media/plimley/TEAM 7B/HTbatch01_pyml/MultiAngle_HT_100_11_py.h5'
fn = 'pix10_5noise0'

f = h5py.File(filename, 'r')

energy_thresh = 300     # keV
tracklist = []

for evtkey in f.keys:
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
