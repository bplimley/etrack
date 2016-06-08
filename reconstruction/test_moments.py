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
import time

from etrack.reconstruction.trackdata import Track
import etrack.io.trackio as trackio
import etrack.reconstruction.trackmoments as tm
import etrack.visualization.trackplot as tp


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

    return tracklist

def test_moments_segmentation(tracklist):
    """
    Check the bounding box for trackmoments
    """

    print('Testing moments segmentation')
    index_error_count = 0
    notimplemented = 0
    for i in range(len(tracklist)):
        t = tracklist[i]
        try:
            mom = tm.MomentsReconstruction(t.image)
            mom.reconstruct()
        except IndexError:
            index_error_count += 1
            continue
        except NotImplementedError:
            notimplemented += 1
            continue

        fig = tp.plot_moments_segment(mom.original_image_kev, mom.box)
        titlestr = '#{}, rough_est={}*, start={}, end={}'.format(
            i, mom.rough_est * 180 / np.pi,
            mom.start_coordinates, mom.end_coordinates)
        plt.title(titlestr)
        plt.show()
        # time.sleep(5)
        plt.close()

    print('{} IndexErrors out of {} tracks'.format(
        index_error_count, len(tracklist)))
    print('{} NotImplementedErrors out of {} tracks'.format(
        notimplemented, len(tracklist)))


filename = '/media/plimley/TEAM 7B/HTbatch01_pyml/MultiAngle_HT_100_11_py.h5'
fn = 'pix10_5noise0'

energy_thresh = 300     # keV
print('Compiling tracklist (energy_thresh = {} keV)'.format(energy_thresh))
tracklist = tracklist_from_h5(filename, energy_thresh)

test_moments_segmentation(tracklist)
