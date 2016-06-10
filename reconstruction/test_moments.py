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
import datetime
import glob

from etrack.reconstruction.trackdata import Track
import etrack.io.trackio as trackio
import etrack.reconstruction.trackmoments as tm
import etrack.visualization.trackplot as tp


def tracklist_from_h5(filename, energy_thresh):
    """
    Go through a pyml h5 file and make Tracks from pix10_5noise0noise0.
    Save them into a list if their energy is above energy_thresh_kev.
    """

    f = h5py.File(filename, 'r')
    fn = 'pix10_5noise0'
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

        tp.plot_moments_segment(mom.original_image_kev, mom.box_x, mom.box_y)
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


def momentlist_from_tracklist(tracklist):
    """
    get lists of each moment
    """

    max_length = 10000

    first_moments = np.zeros((max_length, 2, 2))
    central_moments = np.zeros((max_length, 4, 4))
    rotated_moments = np.zeros((max_length, 4, 4))
    R = np.zeros(max_length)
    phi = np.zeros(max_length)

    n = 0
    index_error_count = 0
    not_implemented_error_count = 0

    print('Performing moments reconstruction...')
    for t in tracklist:
        try:
            mom = tm.MomentsReconstruction(t.image)
            mom.reconstruct()
        except IndexError:
            index_error_count += 1
            continue
        except NotImplementedError:
            not_implemented_error_count += 1
            continue

        # copy R, phi
        R[n] = mom.R
        phi[n] = mom.phi

        # copy moments
        for i in xrange(3):
            for j in xrange(3):
                if i + j <= 1:
                    first_moments[n, i, j] = mom.first_moments[i, j]
                if i + j <= 3:
                    central_moments[n, i, j] = mom.central_moments[i, j]
                    rotated_moments[n, i, j] = mom.rotated_moments[i, j]
        n += 1

    # remove extra entries
    R.resize(n)
    phi.resize(n)
    first_moments = first_moments.copy()
    first_moments.resize((n, 2, 2))
    central_moments = central_moments.copy()
    central_moments.resize((n, 4, 4))
    rotated_moments = rotated_moments.copy()
    rotated_moments.resize((n, 4, 4))

    return first_moments, central_moments, rotated_moments, R, phi


def main1():
    """
    plot the bounding box for segmentation
    """

    tracklist = get_tracklist(n_files=1)

    test_moments_segmentation(tracklist)


def get_tracklist(n_files=10):

    filename = '/media/plimley/TEAM 7B/HTbatch01_pyml/MultiAngle_HT_*_*_py.h5'
    flist = glob.glob(filename)
    energy_thresh = 300     # keV
    tracklist = []

    # get a bunch of tracks: ~250 in energy window per file, ~100 without error
    for fname in flist[:n_files]:
        # add items to list. easy but potentially slow way
        print('Getting tracks from {} at {}...'.format(
            fname, datetime.datetime.now()))
        tracklist += tracklist_from_h5(fname, energy_thresh)

    return tracklist


def main2():
    """
    get the moments lists for histogramming
    """

    tracklist = get_tracklist(n_files=1)

    # get moments from easy ones
    # first, central, rotated = momentlist_from_tracklist(tracklist)
    return momentlist_from_tracklist(tracklist)

    #


def arc_test():
    """
    Use trackmoments.generate_arc() to make a bunch of arcs, and then see
    how well algorithm does at reconstructing.
    """

    n = 10
    rough_est = np.zeros(n)
    alpha = np.zeros(n)
    R = np.zeros(n)
    phi = np.zeros(n)

    print('ideal: pure points, phi=90, r=6')
    center_angles = np.linspace(0, 360, n, endpoint=False)
    for i in xrange(n):
        arc, est = tm.generate_arc(
            r=6, phi_d=90, center_angle_d=center_angles[i], n_pts=1000,
            blur_sigma=0, pixelize=False)
        mom = tm.MomentsReconstruction.reconstruct_arc(arc, est)
        rough_est[i] = mom.rough_est * 180 / np.pi
        alpha[i] = mom.alpha * 180 / np.pi
        R[i] = mom.R
        phi[i] = mom.phi * 180 / np.pi      # degrees now
    print('True directions: {}'.format(rough_est))
    print('Calc directions: {}'.format(alpha))
    print('R: {}'.format(R))
    print('phi: {}'.format(phi))
    print('')

    print('ideal, larger radius: pure points, phi=10, r=30')
    center_angles = np.linspace(0, 360, n, endpoint=False)
    for i in xrange(n):
        arc, est = tm.generate_arc(
            r=30, phi_d=10, center_angle_d=center_angles[i], n_pts=1000,
            blur_sigma=0, pixelize=False)
        mom = tm.MomentsReconstruction.reconstruct_arc(arc, est)
        rough_est[i] = mom.rough_est * 180 / np.pi
        alpha[i] = mom.alpha * 180 / np.pi
        R[i] = mom.R
        phi[i] = mom.phi * 180 / np.pi      # degrees now
    print('True directions: {}'.format(rough_est))
    print('Calc directions: {}'.format(alpha))
    print('R: {}'.format(R))
    print('phi: {}'.format(phi))
    print('')

    print('blurred: sigma = 1. phi=90, r=6')
    center_angles = np.linspace(0, 360, n, endpoint=False)
    for i in xrange(n):
        arc, est = tm.generate_arc(
            r=6, phi_d=90, center_angle_d=center_angles[i], n_pts=1000,
            blur_sigma=1, pixelize=False)
        mom = tm.MomentsReconstruction.reconstruct_arc(arc, est)
        rough_est[i] = mom.rough_est * 180 / np.pi
        alpha[i] = mom.alpha * 180 / np.pi
        R[i] = mom.R
        phi[i] = mom.phi * 180 / np.pi      # degrees now
    print('True directions: {}'.format(rough_est))
    print('Calc directions: {}'.format(alpha))
    print('R: {}'.format(R))
    print('phi: {}'.format(phi))
    print('')

    print('blurred and pixelized: sigma = 1. phi=90, r=6')
    center_angles = np.linspace(0, 360, n, endpoint=False)
    for i in xrange(n):
        arc, est = tm.generate_arc(
            r=6, phi_d=90, center_angle_d=center_angles[i], n_pts=1000,
            blur_sigma=1, pixelize=True)
        mom = tm.MomentsReconstruction.reconstruct_arc(arc, est)
        rough_est[i] = mom.rough_est * 180 / np.pi
        alpha[i] = mom.alpha * 180 / np.pi
        R[i] = mom.R
        phi[i] = mom.phi * 180 / np.pi      # degrees now
    print('True directions: {}'.format(rough_est))
    print('Calc directions: {}'.format(alpha))
    print('R: {}'.format(R))
    print('phi: {}'.format(phi))
    print('')

    print('blurred and pixelized: sigma = 1. phi=10, r=30')
    center_angles = np.linspace(0, 360, n, endpoint=False)
    for i in xrange(n):
        arc, est = tm.generate_arc(
            r=30, phi_d=10, center_angle_d=center_angles[i], n_pts=1000,
            blur_sigma=1, pixelize=True)
        mom = tm.MomentsReconstruction.reconstruct_arc(arc, est)
        rough_est[i] = mom.rough_est * 180 / np.pi
        alpha[i] = mom.alpha * 180 / np.pi
        R[i] = mom.R
        phi[i] = mom.phi * 180 / np.pi      # degrees now
    print('True directions: {}'.format(rough_est))
    print('Calc directions: {}'.format(alpha))
    print('R: {}'.format(R))
    print('phi: {}'.format(phi))
    print('')

    print('blurred and pixelized: sigma = 1. phi=10, r=30, n_pts = 10000')
    center_angles = np.linspace(0, 360, n, endpoint=False)
    for i in xrange(n):
        arc, est = tm.generate_arc(
            r=30, phi_d=10, center_angle_d=center_angles[i], n_pts=10000,
            blur_sigma=1, pixelize=True)
        mom = tm.MomentsReconstruction.reconstruct_arc(arc, est)
        rough_est[i] = mom.rough_est * 180 / np.pi
        alpha[i] = mom.alpha * 180 / np.pi
        R[i] = mom.R
        phi[i] = mom.phi * 180 / np.pi      # degrees now
    print('True directions: {}'.format(rough_est))
    print('Calc directions: {}'.format(alpha))
    print('R: {}'.format(R))
    print('phi: {}'.format(phi))
    print('')


if __name__ == '__main__':
    main2()
