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
    do moments reconstruction
    """

    momlist = []

    print('Performing moments reconstruction...')
    t0 = time.time()
    for t in tracklist:
        mom = tm.MomentsReconstruction(t.image)
        mom.reconstruct()
        momlist.append(mom)
    t1 = time.time()
    print('Reconstructed {} tracks in {} s ({} s/track)'.format(
        len(tracklist), t1 - t0, (t1 - t0) / len(tracklist)))
    return momlist


def moments_from_momentlist(momentlist):
    """
    Pull out relevant variables (moments, R, phi, etc.) from a list of objects.
    """

    max_length = 10000

    first_moments = np.zeros((max_length, 2, 2))
    central_moments = np.zeros((max_length, 4, 4))
    rotated_moments = np.zeros((max_length, 4, 4))
    R = np.zeros(max_length)
    phi = np.zeros(max_length)
    arclength = np.zeros(max_length)
    pr3a = np.zeros(max_length)
    pr3b = np.zeros(max_length)

    n = 0

    for mom in momentlist:
        # copy R, phi
        R[n] = mom.R
        phi[n] = mom.phi
        arclength[n] = mom.arclength
        pr3a[n] = mom.pathology_ratio_3a
        pr3b[n] = mom.pathology_ratio_3b

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
    arclength.resize(n)
    pr3a.resize(n)
    pr3b.resize(n)
    first_moments = first_moments.copy()
    first_moments.resize((n, 2, 2))
    central_moments = central_moments.copy()
    central_moments.resize((n, 4, 4))
    rotated_moments = rotated_moments.copy()
    rotated_moments.resize((n, 4, 4))

    moment_vars = (first_moments, central_moments, rotated_moments,
                   R, phi, arclength, pr3a, pr3b)
    return moment_vars


def plot_track_arc(track, debug=False, end_segment=False, box=False,
                   title=None):
    mom = tm.MomentsReconstruction(track.image)
    mom.reconstruct()
    tp.plot_moments_arc(
        mom, debug=debug, end_segment=end_segment, box=box, title=title)


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


def main3(tracklist=None, mlist=None):
    """
    Run moments algorithm on tracks, get delta-alpha, and plot histograms.
    """

    nf = 1
    binwidth = 5    # degrees
    if tracklist is None and mlist is None:
        tracklist = get_tracklist(n_files=nf)
    if mlist is None:
        # run moments algorithm
        mlist = momentlist_from_tracklist(tracklist)

    # get delta alpha
    da = np.array(
        [mlist[i].alpha * 180 / np.pi - tracklist[i].g4track.alpha_deg
         for i in xrange(len(tracklist))])
    while np.any(da > 180):
        da[da > 180] -= 360
    while np.any(da < -180):
        da[da < -180] += 360

    # get moments
    moment_vars = moments_from_momentlist(mlist)
    first, central, rotated, R, phi, arclen, pr3a, pr3b = moment_vars

    if False:
        # total da histogram
        plt.figure()
        n, bins = np.histogram(da, np.arange(-180, 180.1, binwidth))
        plt.plot(bins[:-1] + binwidth / 2, n.astype(np.float) / np.sum(n),
                 'k', lw=2, drawstyle='steps-mid', label='all')
        # da for
        lg = np.abs(phi) < 90. / 180 * np.pi
        n, bins = np.histogram(da[lg], np.arange(-180, 180.1, binwidth))
        plt.plot(bins[:-1] + binwidth / 2, n.astype(np.float) / np.sum(n),
                 'b', lw=2, drawstyle='steps-mid', label='phi < 90')
        # da for
        lg = np.abs(phi) < 45. / 180 * np.pi
        n, bins = np.histogram(da[lg], np.arange(-180, 180.1, binwidth))
        plt.plot(bins[:-1] + binwidth / 2, n.astype(np.float) / np.sum(n),
                 'm', lw=2, drawstyle='steps-mid', label='phi < 45')
        # da for
        lg = np.abs(phi) < 15. / 180 * np.pi
        n, bins = np.histogram(da[lg], np.arange(-180, 180.1, binwidth))
        plt.plot(bins[:-1] + binwidth / 2, n.astype(np.float) / np.sum(n),
                 'g', lw=2, drawstyle='steps-mid', label='phi < 15')
        plt.xlim([-180, 180])
        plt.xlabel('Delta Alpha [degrees]')
        plt.ylabel('fraction of tracks per {} degrees'.format(binwidth))
        plt.title('Moments method')
        plt.legend()
        plt.show()

    lg0 = np.abs(da) < 8
    lg1 = np.abs(da) < 20

    if False:
        # total arclen histogram
        binwidth = 0.25
        plt.figure()
        n, bins = np.histogram(arclen, np.arange(0, 20, binwidth))
        plt.plot(bins[:-1] + binwidth / 2, n.astype(np.float) / np.sum(n),
                 'k', drawstyle='steps-mid', label='all')
        # lg1
        n, bins = np.histogram(arclen[lg1], np.arange(0, 20, binwidth))
        plt.plot(bins[:-1] + binwidth / 2, n.astype(np.float) / np.sum(n),
                 'r', drawstyle='steps-mid', label='|da| < 20 degrees')
        # lg0
        n, bins = np.histogram(arclen[lg0], np.arange(0, 20, binwidth))
        plt.plot(bins[:-1] + binwidth / 2, n.astype(np.float) / np.sum(n),
                 'c', drawstyle='steps-mid', label='|da| < 8 degrees')
        plt.xlabel('Arc length [pixels]')
        plt.ylabel('fraction of tracks per {} arclength'.format(binwidth))
        plt.legend()
        plt.show()

    if False:
        # total phi histogram
        binwidth = 5
        plt.figure()
        n, bins = np.histogram(phi * 180 / np.pi, np.arange(0, 360, binwidth))
        plt.plot(bins[:-1] + binwidth / 2, n.astype(np.float) / np.sum(n),
                 'k', drawstyle='steps-mid', label='all')
        # lg1
        n, bins = np.histogram(
            phi[lg1] * 180 / np.pi, np.arange(0, 360, binwidth))
        plt.plot(bins[:-1] + binwidth / 2, n.astype(np.float) / np.sum(n),
                 'r', drawstyle='steps-mid', label='|da| < 20 degrees')
        # lg0
        n, bins = np.histogram(
            phi[lg0] * 180 / np.pi, np.arange(0, 360, binwidth))
        plt.plot(bins[:-1] + binwidth / 2, n.astype(np.float) / np.sum(n),
                 'c', drawstyle='steps-mid', label='|da| < 8 degrees')
        plt.xlabel('Phi [degrees]')
        plt.ylabel('fraction of tracks per {} degrees'.format(binwidth))
        plt.legend()
        plt.show()

    if False:
        # pr3a histogram
        binwidth = 0.025
        plt.figure()
        n, bins = np.histogram(np.abs(pr3a), np.arange(0, 4, binwidth))
        plt.plot(bins[:-1] + binwidth / 2, n.astype(np.float) / np.sum(n),
                 'k', drawstyle='steps-mid', label='all')
        # lg1
        n, bins = np.histogram(np.abs(pr3a)[lg1], np.arange(0, 4, binwidth))
        plt.plot(bins[:-1] + binwidth / 2, n.astype(np.float) / np.sum(n),
                 'r', drawstyle='steps-mid', label='|da| < 20 degrees')
        # lg0
        n, bins = np.histogram(np.abs(pr3a)[lg0], np.arange(0, 4, binwidth))
        plt.plot(bins[:-1] + binwidth / 2, n.astype(np.float) / np.sum(n),
                 'c', drawstyle='steps-mid', label='|da| < 8 degrees')
        plt.xlabel('T12 / T21')
        plt.ylabel('fraction of tracks per {} ratio'.format(binwidth))
        plt.legend()
        plt.show()

    if True:
        # pr3b histogram
        binwidth = 0.0025
        plt.figure()
        n, bins = np.histogram(np.abs(1 / pr3b), np.arange(0, 1, binwidth))
        plt.plot(bins[:-1] + binwidth / 2, n.astype(np.float) / np.sum(n),
                 'k', drawstyle='steps-mid', label='all')
        # lg1
        n, bins = np.histogram(np.abs(1 / pr3b)[lg1], np.arange(0, 1, binwidth))
        plt.plot(bins[:-1] + binwidth / 2, n.astype(np.float) / np.sum(n),
                 'r', drawstyle='steps-mid', label='|da| < 20 degrees')
        # lg0
        n, bins = np.histogram(np.abs(1 / pr3b)[lg0], np.arange(0, 1, binwidth))
        plt.plot(bins[:-1] + binwidth / 2, n.astype(np.float) / np.sum(n),
                 'c', drawstyle='steps-mid', label='|da| < 8 degrees')
        plt.xlabel('T03 / T30')
        plt.ylabel('fraction of tracks per {} ratio'.format(binwidth))
        plt.legend()
        plt.show()

    return moment_vars


def main4(tracklist=None, HTalpha=None, mlist=None):
    """
    Plot HybridTrack results, and compare to moments.
    """

    # get moments
    moment_vars = moments_from_momentlist(mlist)
    first, central, rotated, R, phi, arclen, pr3a, pr3b = moment_vars

    HT_da = np.array(
        [HTalpha[i] - tracklist[i].g4track.alpha_deg
         for i in xrange(len(tracklist))])
    while np.any(HT_da > 180):
        HT_da[HT_da > 180] -= 360
    while np.any(HT_da < -180):
        HT_da[HT_da < -180] += 360

    MR_da = np.array(
        [mlist[i].alpha * 180 / np.pi - tracklist[i].g4track.alpha_deg
         for i in xrange(len(tracklist))])
    while np.any(MR_da > 180):
        MR_da[MR_da > 180] -= 360
    while np.any(MR_da < -180):
        MR_da[MR_da < -180] += 360

    # da histograms

    # full
    binwidth = 3
    plt.figure()
    n, bins = np.histogram(MR_da, np.arange(-180, 180.1, binwidth))
    plt.plot(bins[:-1] + binwidth / 2, n.astype(np.float) / np.sum(n),
             'k', lw=2, drawstyle='steps-mid', label='moments')
    n, bins = np.histogram(HT_da, np.arange(-180, 180.1, binwidth))
    plt.plot(bins[:-1] + binwidth / 2, n.astype(np.float) / np.sum(n),
             'b', lw=2, drawstyle='steps-mid', label='ridge-following')
    plt.xlim([-180, 180])
    plt.ylim([0, 0.12])
    plt.xlabel('Delta Alpha [degrees]')
    plt.ylabel('fraction of tracks per {} degrees'.format(binwidth))
    plt.title('All tracks in set (E > 300 keV)')
    plt.legend()
    plt.show()

    # phi < 90
    plt.figure()
    lg = np.abs(phi) < 90. / 180 * np.pi
    n, bins = np.histogram(MR_da[lg], np.arange(-180, 180.1, binwidth))
    plt.plot(bins[:-1] + binwidth / 2, n.astype(np.float) / np.sum(n),
             'k', lw=2, drawstyle='steps-mid', label='moments')
    n, bins = np.histogram(HT_da[lg], np.arange(-180, 180.1, binwidth))
    plt.plot(bins[:-1] + binwidth / 2, n.astype(np.float) / np.sum(n),
             'b', lw=2, drawstyle='steps-mid',
             label='ridge-following')
    plt.xlim([-180, 180])
    plt.ylim([0, 0.12])
    plt.xlabel('Delta Alpha [degrees]')
    plt.ylabel('fraction of tracks per {} degrees'.format(binwidth))
    plt.title('Phi < 90 degrees (E > 300 keV)')
    plt.legend()
    plt.show()

    # phi < 45
    plt.figure()
    lg = np.abs(phi) < 45. / 180 * np.pi
    n, bins = np.histogram(MR_da[lg], np.arange(-180, 180.1, binwidth))
    plt.plot(bins[:-1] + binwidth / 2, n.astype(np.float) / np.sum(n),
             'k', lw=2, drawstyle='steps-mid', label='moments')
    n, bins = np.histogram(HT_da[lg], np.arange(-180, 180.1, binwidth))
    plt.plot(bins[:-1] + binwidth / 2, n.astype(np.float) / np.sum(n),
             'b', lw=2, drawstyle='steps-mid',
             label='ridge-following')
    plt.xlim([-180, 180])
    plt.ylim([0, 0.12])
    plt.xlabel('Delta Alpha [degrees]')
    plt.ylabel('fraction of tracks per {} degrees'.format(binwidth))
    plt.title('Phi < 45 degrees (E > 300 keV)')
    plt.legend()
    plt.show()

    # phi < 15
    plt.figure()
    lg = np.abs(phi) < 15. / 180 * np.pi
    n, bins = np.histogram(MR_da[lg], np.arange(-180, 180.1, binwidth))
    plt.plot(bins[:-1] + binwidth / 2, n.astype(np.float) / np.sum(n),
             'k', lw=2, drawstyle='steps-mid', label='moments')
    n, bins = np.histogram(HT_da[lg], np.arange(-180, 180.1, binwidth))
    plt.plot(bins[:-1] + binwidth / 2, n.astype(np.float) / np.sum(n),
             'b', lw=2, drawstyle='steps-mid',
             label='ridge-following')
    plt.xlim([-180, 180])
    plt.ylim([0, 0.12])
    plt.xlabel('Delta Alpha [degrees]')
    plt.ylabel('fraction of tracks per {} degrees'.format(binwidth))
    plt.title('Phi < 15 degrees (E > 300 keV)')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main2()
