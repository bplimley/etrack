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
import os
import collections
from datetime import datetime as dt

from etrack.reconstruction.trackdata import Track
import etrack.io.trackio as trackio
import etrack.reconstruction.trackmoments as tm
import etrack.reconstruction.evaluation as ev
import etrack.reconstruction.classify as cl
import etrack.visualization.trackplot as tp

DEG = '$^\circ$'


def tracks_for_don(momlist, tracklist, classifierlist):
    """
    Moment reconstruction evaluation set for Don Gunter. 6/23/2016.

    For 1000 tracks:
    - make a dual-pane image showing true and computed directions, etc.
    - record parameters into a CSV file.
    """

    nmax = 1000
    savedir = '/media/plimley/TEAM 7B/tracks_for_Don_5/'
    csv_name = 'parameters.csv'

    make_figures = False

    if make_figures:
        # figure images
        print('Making {} figures at {}...'.format(nmax, dt.now()))
        for i in xrange(nmax):
            titlestr = '{}, b={:.0f}{deg}, angle={:.1f}{deg}'.format(
                i,
                classifierlist[i].g4track.beta_deg,
                classifierlist[i].total_scatter_angle * 180 / np.pi,
                deg=DEG)
            if np.abs(classifierlist[i].g4track.beta_deg) > 60:
                titlestr += ' [Beta > 60{}]'.format(DEG)
            if classifierlist[i].early_scatter:
                titlestr += ' [Early scatter in 25um]'
            if classifierlist[i].overlap:
                titlestr += ' [Overlapping]'
            if classifierlist[i].wrong_end:
                titlestr += ' [Wrong end]'
            # if momlist[i] is None:
            #     titlestr += ' [Failed edge check!]'
            # elif np.isnan(momlist[i].rotation_angle):
            #     titlestr += ' [Failed edge check!]'
            # elif np.isnan(momlist[i].R):
            #     titlestr += ' [bad radius calculation]'
            # if momlist[i].edge_pixel_segments > 1:
            #     titlestr += ' [Warning: edge segments > 1]'
            # if momlist[i].edge_pixel_count > 4:
            #     titlestr += ' [Warning: edge pixels > 4]'
            # if momlist[i].end_energy > 25:
            #     titlestr += ' [Warning: end energy > 25keV]'
            # if momlist[i].phi > 1:
            #     titlestr += ' [Warning: phi > 1 rad]'

            f = tp.plot_moments_track(momlist[i], tracklist[i], title='')
            f.suptitle(titlestr)
            plt.show()
            fname = '{0:03d}.png'.format(i)
            fpath = os.path.join(savedir, fname)
            f.savefig(fpath, format='png', bbox_inches='tight', pad_inches=0.5)
            plt.close(f)

    # parameters for CSV file
    print('Collecting parameters at {}...'.format(dt.now()))
    params = collections.OrderedDict()
    params['id'] = ['{0:03d}'.format(i) for i in xrange(nmax)]
    params['E'] = np.array([t.energy_kev for t in tracklist[:nmax]])
    params['beta'] = [c.g4track.beta_deg for c in classifierlist[:nmax]]
    da = np.array(
        [momlist[i].alpha * 180 / np.pi - tracklist[i].g4track.alpha_deg
         for i in xrange(nmax)])
    while np.any(da > 180):
        da[da > 180] -= 360
    while np.any(da < -180):
        da[da < -180] += 360
    params['delta_alpha_deg'] = da
    params['early_scatter'] = [
        c.early_scatter + 0 for c in classifierlist[:nmax]]
    params['scatter_angle'] = [
        c.total_scatter_angle * 180 / np.pi for c in classifierlist[:nmax]]
    params['overlap'] = [c.overlap + 0 for c in classifierlist[:nmax]]
    params['wrong_end'] = [c.wrong_end + 0 for c in classifierlist[:nmax]]
    params['edge_pixel_count'] = np.array(
        [m.edge_pixel_count for m in momlist[:nmax]])
    params['edge_pixel_segments'] = np.array(
        [m.edge_pixel_segments for m in momlist[:nmax]])
    params['end_energy'] = np.array([m.end_energy for m in momlist[:nmax]])
    params['edge_avg_dist'] = np.array(
        [m.edge_avg_dist for m in momlist[:nmax]])
    params['phi'] = np.array([m.phi for m in momlist[:nmax]])
    params['R'] = np.array([m.R for m in momlist[:nmax]])
    params['rotation_angle'] = np.array(
        [m.rotation_angle for m in momlist[:nmax]])
    params['T12/T21'] = np.array(
        [m.pathology_ratio_3a for m in momlist[:nmax]])
    params['T30/T03'] = np.array(
        [m.pathology_ratio_3b for m in momlist[:nmax]])
    TR = np.array([m.rotated_moments for m in momlist[:nmax]])
    TC = np.array([m.central_moments for m in momlist[:nmax]])
    T0 = np.array([m.first_moments for m in momlist[:nmax]])
    ij_list = (
        (0, 0),
        (1, 0), (0, 1),
        (2, 0), (1, 1), (0, 2),
        (3, 0), (2, 1), (1, 2), (0, 3))
    for i, j in ij_list:
        key = 'T{}{}_R'.format(i, j)
        params[key] = TR[:, i, j].flatten()
    for i, j in ij_list:
        key = 'T{}{}_C'.format(i, j)
        params[key] = TC[:, i, j].flatten()
    for i, j in ij_list[:3]:
        key = 'T{}{}_0'.format(i, j)
        params[key] = T0[:, i, j].flatten()

    header = ','.join(['{}' for _ in xrange(len(params))]).format(
        *params.keys()) + '\n'

    print('Generating lines for data file at {}...'.format(dt.now()))
    datalines = []
    for i in xrange(nmax):
        these_params = collections.OrderedDict()
        for k in params.keys():
            these_params[k] = params[k][i]
        datalines.append(','.join(['{}' for _ in xrange(len(params))]).format(
            *these_params.values()) + '\n')

    print('Writing file at {}...'.format(dt.now()))
    with open(os.path.join(savedir, csv_name), 'w') as fobj:
        fobj.writelines([header])
        fobj.writelines(datalines)

    print('Done! at {}'.format(dt.now()))


def tracklist_from_h5(filename, energy_thresh):
    """
    Go through a pyml h5 file and make Tracks from pix10_5noise0.
    Save them into a list if their energy is above energy_thresh_kev.
    """

    try:
        f = h5py.File(filename, 'r')
    except IOError:
        print('!!! IOError on {} !!!'.format(filename))
        return []
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

    for i in range(len(tracklist)):
        t = tracklist[i]
        mom = tm.MomentsReconstruction(t.image)
        mom.reconstruct()

        tp.plot_moments_segment(mom)
        titlestr = '#{}, rough_est={}*, start={}, end={}'.format(
            i, mom.rough_est * 180 / np.pi,
            mom.start_coordinates, mom.end_coordinates)
        plt.title(titlestr)


def momentlist_from_tracklist(tracklist):
    """
    do moments reconstruction
    """

    momlist = []

    print('Performing moments reconstruction...')
    t0 = time.time()
    for t in tracklist:
        mom = tm.MomentsReconstruction(t.image)
        mom.track = t
        try:
            mom.reconstruct()
        except (tm.CheckSegmentBoxError, RuntimeError):
            # fill in nan's for everything that tracks_for_don wants
            mom.alpha = np.nan
            mom.phi = np.nan
            mom.R = np.nan
            mom.rotation_angle = np.nan
            mom.pathology_ratio_3a = np.nan
            mom.pathology_ratio_3b = np.nan
            mom.rotated_moments = np.empty((4, 4)) * np.nan
            mom.central_moments = np.empty((4, 4)) * np.nan
            mom.first_moments = np.empty((2, 2)) * np.nan
            mom.edge_pixel_count = np.nan
            mom.edge_pixel_segments = np.nan
            mom.edge_avg_dist = np.nan
            mom.ends_energy = np.nan
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
    z = np.zeros(max_length)
    E = np.zeros(max_length)
    end_energy = np.zeros(max_length)

    n = 0

    for mom in momentlist:
        # copy R, phi
        R[n] = mom.R
        phi[n] = mom.phi
        arclength[n] = mom.Rphi
        pr3a[n] = mom.pathology_ratio_3a
        pr3b[n] = mom.pathology_ratio_3b
        z[n] = mom.track.g4track.x0[-1]     # -0.65 to 0
        E[n] = mom.track.g4track.energy_tot_kev
        end_energy[n] = mom.end_energy

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
    z.resize(n)
    E.resize(n)
    end_energy.resize(n)
    first_moments = first_moments.copy()
    first_moments.resize((n, 2, 2))
    central_moments = central_moments.copy()
    central_moments.resize((n, 4, 4))
    rotated_moments = rotated_moments.copy()
    rotated_moments.resize((n, 4, 4))

    moment_vars = (first_moments, central_moments, rotated_moments,
                   R, phi, arclength, pr3a, pr3b, z, E, end_energy)
    return moment_vars


def classifierlist_from_tracklist(tracklist, momlist, classify=True):
    """
    Make a list of Classifier objects from a tracklist and momentslist.

    The momentslist is for checking the end.

    classify=True: perform the classification. (default)
    classify=False: create the objects but do not classify.
    """

    classifierlist = [cl.Classifier(t.g4track) for t in tracklist]

    if classify:
        for i, c in enumerate(classifierlist):
            try:
                c.mc_classify()
            except cl.TrackTooShortError:
                c.error = 'TrackTooShortError'
            else:
                if momlist is not None:
                    try:
                        c.end_classify(tracklist[i], mom=momlist[i])
                    except tp.G4TrackTooBigError:
                        c.error = 'G4TrackTooBigError'

    return classifierlist


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


def get_tracklist(n_files=8, energy_thresh=0):

    filename = '/media/plimley/TEAM 7B/HTbatch01_pyml/MultiAngle_HT_*_*_py.h5'
    flist = glob.glob(filename)
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

    # z = -0.65 is the pixel plane (narrow tracks)
    # z = 0 is the back plane (diffuse tracks)
    zmin = -0.65
    zmax = -0.0

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
    # moment_vars = moments_from_momentlist(mlist)
    # first, central, rotated, R, phi, arclen, pr3a, pr3b, z, E = moment_vars
    phi = np.array([mom.phi for mom in mlist])
    edge_pixel_count = np.array([mom.edge_pixel_count for mom in mlist])
    end_energy = np.array([mom.end_energy for mom in mlist])

    # depth selection
    # lgdepth = (z >= zmin) & (z <= zmax)

    # da = da[lgdepth]
    # R = R[lgdepth]
    # phi = phi[lgdepth]
    # arclen = arclen[lgdepth]
    # pr3a = pr3a[lgdepth]
    # pr3b = pr3b[lgdepth]
    # z = z[lgdepth]
    # E = E[lgdepth]
    # haven't selected the raw moments yet (first, central, rotated)

    if True:
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

    if True:
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

    if False:
        # pr3b histogram
        binwidth = 0.0025
        plt.figure()
        n, bins = np.histogram(np.abs(1 / pr3b), np.arange(0, 1, binwidth))
        plt.plot(bins[:-1] + binwidth / 2, n.astype(np.float) / np.sum(n),
                 'k', drawstyle='steps-mid', label='all')
        # lg1
        n, bins = np.histogram(
            np.abs(1 / pr3b)[lg1], np.arange(0, 1, binwidth))
        plt.plot(bins[:-1] + binwidth / 2, n.astype(np.float) / np.sum(n),
                 'r', drawstyle='steps-mid', label='|da| < 20 degrees')
        # lg0
        n, bins = np.histogram(
            np.abs(1 / pr3b)[lg0], np.arange(0, 1, binwidth))
        plt.plot(bins[:-1] + binwidth / 2, n.astype(np.float) / np.sum(n),
                 'c', drawstyle='steps-mid', label='|da| < 8 degrees')
        plt.xlabel('T03 / T30')
        plt.ylabel('fraction of tracks per {} ratio'.format(binwidth))
        plt.legend()
        plt.show()

    if False:
        # new for version 3: edge pixel count
        binwidth = 1.0
        plt.figure()
        n, bins = np.histogram(edge_pixel_count, np.arange(0, 20, binwidth))
        plt.plot(bins[:-1] + binwidth / 2, n.astype(np.float) / np.sum(n),
                 'k', drawstyle='steps-mid', label='all')
        # lg1
        n, bins = np.histogram(
            edge_pixel_count[lg1], np.arange(0, 20, binwidth))
        plt.plot(bins[:-1] + binwidth / 2, n.astype(np.float) / np.sum(n),
                 'r', drawstyle='steps-mid', label='|da| < 20 degrees')
        # lg0
        n, bins = np.histogram(
            edge_pixel_count[lg0], np.arange(0, 20, binwidth))
        plt.plot(bins[:-1] + binwidth / 2, n.astype(np.float) / np.sum(n),
                 'c', drawstyle='steps-mid', label='|da| < 8 degrees')
        plt.xlabel('edge_pixel_count')
        plt.ylabel('fraction of tracks per {} pixel count'.format(binwidth))
        plt.legend()
        plt.show()

    if True:
        # new for version 3: end energy
        binwidth = 1.0
        plt.figure()
        n, bins = np.histogram(end_energy, np.arange(0, 50, binwidth))
        plt.plot(bins[:-1] + binwidth / 2, n.astype(np.float) / np.sum(n),
                 'k', drawstyle='steps-mid', label='all')
        # lg1
        n, bins = np.histogram(
            end_energy[lg1], np.arange(0, 50, binwidth))
        plt.plot(bins[:-1] + binwidth / 2, n.astype(np.float) / np.sum(n),
                 'r', drawstyle='steps-mid', label='|da| < 20 degrees')
        # lg0
        n, bins = np.histogram(
            end_energy[lg0], np.arange(0, 50, binwidth))
        plt.plot(bins[:-1] + binwidth / 2, n.astype(np.float) / np.sum(n),
                 'c', drawstyle='steps-mid', label='|da| < 8 degrees')
        plt.xlabel('end energy [keV]')
        plt.ylabel('fraction of tracks per {} keV'.format(binwidth))
        plt.legend()
        plt.show()

    # return moment_vars


def main4(tracklist=None, HTalpha=None, mlist=None):
    """
    Plot HybridTrack results, and compare to moments.
    """

    # get moments
    # moment_vars = moments_from_momentlist(mlist)
    # first, central, rotated, R, phi, arclen, pr3a, pr3b = moment_vars
    phi = np.array([mom.phi for mom in mlist])

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


def main5(momlist1, momlist2, HTalpha, tracklist):
    """
    Compare two versions of the moments method with the ridge following.
    """

    # relevant indicators
    phi1 = np.array([mom.phi for mom in momlist1])
    phi2 = np.array([mom.phi for mom in momlist2])

    end_energy2 = np.array([mom.end_energy for mom in momlist2])
    edge_pixel_count2 = np.array([mom.edge_pixel_count for mom in momlist2])
    edge_pixel_segments2 = np.array(
        [mom.edge_pixel_segments for mom in momlist2])

    # generate and correct delta alpha's
    HT_da = np.array(
        [HTalpha[i] - tracklist[i].g4track.alpha_deg
         for i in xrange(len(tracklist))])
    while np.any(HT_da > 180):
        HT_da[HT_da > 180] -= 360
    while np.any(HT_da < -180):
        HT_da[HT_da < -180] += 360

    MR1_da = np.array(
        [momlist1[i].alpha * 180 / np.pi - tracklist[i].g4track.alpha_deg
         for i in xrange(len(tracklist))])
    while np.any(MR1_da > 180):
        MR1_da[MR1_da > 180] -= 360
    while np.any(MR1_da < -180):
        MR1_da[MR1_da < -180] += 360

    MR3_da = np.array(
        [momlist2[i].alpha * 180 / np.pi - tracklist[i].g4track.alpha_deg
         for i in xrange(len(tracklist))])
    while np.any(MR3_da > 180):
        MR3_da[MR3_da > 180] -= 360
    while np.any(MR3_da < -180):
        MR3_da[MR3_da < -180] += 360

    binwidth = 3

    # full
    r = np.sum(np.logical_not(np.isnan(MR1_da))) / float(len(MR1_da))
    print('moments v1 success rate: {}'.format(r))
    r = np.sum(np.logical_not(np.isnan(MR3_da))) / float(len(MR3_da))
    print('moments v3 success rate: {}'.format(r))
    r = np.sum(np.logical_not(np.isnan(HT_da))) / float(len(HT_da))
    print('ridge-following success rate: {}'.format(r))

    if True:
        plt.figure()
        n, bins = np.histogram(MR1_da, np.arange(-180, 180.1, binwidth))
        print('sum(n) = {}'.format(np.sum(n)))
        plt.plot(bins[:-1] + binwidth / 2, n.astype(np.float) / np.sum(n),
                 'b', lw=2, drawstyle='steps-mid', label='moments-1')
        n, bins = np.histogram(MR3_da, np.arange(-180, 180.1, binwidth))
        print('sum(n) = {}'.format(np.sum(n)))
        plt.plot(bins[:-1] + binwidth / 2, n.astype(np.float) / np.sum(n),
                 'g', lw=2, drawstyle='steps-mid', label='moments-3')
        n, bins = np.histogram(HT_da, np.arange(-180, 180.1, binwidth))
        print('sum(n) = {}'.format(np.sum(n)))
        plt.plot(bins[:-1] + binwidth / 2, n.astype(np.float) / np.sum(n),
                 'k', lw=2, drawstyle='steps-mid', label='ridge-following')
        plt.xlim([-180, 180])
        plt.ylim([0, 0.12])
        plt.xlabel('Delta Alpha [degrees]')
        plt.ylabel('fraction of tracks per {} degrees'.format(binwidth))
        plt.title('All tracks in set (E > 300 keV)')
        plt.legend()
        plt.show()

    # filter with phi on both moments lists
    lg1 = (np.abs(phi1) < 1.5)
    lg2 = (np.abs(phi2) < 1.5)

    r = np.sum(lg1) / float(len(MR1_da))
    print('moments v1 phi < 1.5 rate: {}'.format(r))
    r = np.sum(lg2) / float(len(MR3_da))
    print('moments v3 phi < 1.5 rate: {}'.format(r))

    if True:
        plt.figure()
        n, bins = np.histogram(MR1_da[lg1], np.arange(-180, 180.1, binwidth))
        print('sum(n) = {}'.format(np.sum(n)))
        plt.plot(bins[:-1] + binwidth / 2, n.astype(np.float) / np.sum(n),
                 'b', lw=2, drawstyle='steps-mid', label='moments-1 filtered')
        n, bins = np.histogram(MR3_da[lg2], np.arange(-180, 180.1, binwidth))
        print('sum(n) = {}'.format(np.sum(n)))
        plt.plot(bins[:-1] + binwidth / 2, n.astype(np.float) / np.sum(n),
                 'g', lw=2, drawstyle='steps-mid', label='moments-3 filtered')
        n, bins = np.histogram(HT_da, np.arange(-180, 180.1, binwidth))
        print('sum(n) = {}'.format(np.sum(n)))
        plt.plot(bins[:-1] + binwidth / 2, n.astype(np.float) / np.sum(n),
                 'k', lw=2, drawstyle='steps-mid',
                 label='ridge-following unfiltered')
        plt.xlim([-180, 180])
        plt.ylim([0, 0.12])
        plt.xlabel('Delta Alpha [degrees]')
        plt.ylabel('fraction of tracks per {} degrees'.format(binwidth))
        plt.title('Filter moments by phi (phi < 1.5 rad; E > 300 keV)')
        plt.legend()
        plt.show()

    # filter with phi and edge segments on moments v3
    lg1 = (np.abs(phi1) < 1.5)
    lg2 = ((np.abs(phi2) < 1.5) & (edge_pixel_segments2 == 1))

    r = np.sum(lg2) / float(len(MR3_da))
    print('moments v3 (phi < 1.5 and edge_pixel_segments = 1) rate: {}'.format(
        r))

    if True:
        plt.figure()
        n, bins = np.histogram(MR1_da[lg1], np.arange(-180, 180.1, binwidth))
        print('sum(n) = {}'.format(np.sum(n)))
        plt.plot(bins[:-1] + binwidth / 2, n.astype(np.float) / np.sum(n),
                 'b', lw=2, drawstyle='steps-mid', label='moments-1 filtered')
        n, bins = np.histogram(MR3_da[lg2], np.arange(-180, 180.1, binwidth))
        print('sum(n) = {}'.format(np.sum(n)))
        plt.plot(bins[:-1] + binwidth / 2, n.astype(np.float) / np.sum(n),
                 'g', lw=2, drawstyle='steps-mid', label='moments-3 filtered')
        n, bins = np.histogram(HT_da, np.arange(-180, 180.1, binwidth))
        print('sum(n) = {}'.format(np.sum(n)))
        plt.plot(bins[:-1] + binwidth / 2, n.astype(np.float) / np.sum(n),
                 'k', lw=2, drawstyle='steps-mid',
                 label='ridge-following unfiltered')
        plt.xlim([-180, 180])
        plt.ylim([0, 0.12])
        plt.xlabel('Delta Alpha [degrees]')
        plt.ylabel('fraction of tracks per {} degrees'.format(binwidth))
        plt.title('Filter moments by phi and edge_pixel_segments ' +
                  '(phi < 1.5 rad; E > 300 keV)')
        plt.legend()
        plt.show()

    # filter with phi and edge segments and pixel count on moments v3
    lg1 = (np.abs(phi1) < 1.5)
    lg2 = (
        (np.abs(phi2) < 1.5) &
        (edge_pixel_segments2 == 1) &
        (edge_pixel_count2 <= 4))

    r = np.sum(lg2) / float(len(MR3_da))
    print('moments v3 (phi < 1.5 and edge_pixel_segments = 1 and ' +
          'edge_pixel_count <= 4) rate: {}'.format(r))

    if True:
        plt.figure()
        n, bins = np.histogram(MR1_da[lg1], np.arange(-180, 180.1, binwidth))
        print('sum(n) = {}'.format(np.sum(n)))
        plt.plot(bins[:-1] + binwidth / 2, n.astype(np.float) / np.sum(n),
                 'b', lw=2, drawstyle='steps-mid', label='moments-1 filtered')
        n, bins = np.histogram(MR3_da[lg2], np.arange(-180, 180.1, binwidth))
        print('sum(n) = {}'.format(np.sum(n)))
        plt.plot(bins[:-1] + binwidth / 2, n.astype(np.float) / np.sum(n),
                 'g', lw=2, drawstyle='steps-mid', label='moments-3 filtered')
        n, bins = np.histogram(HT_da, np.arange(-180, 180.1, binwidth))
        print('sum(n) = {}'.format(np.sum(n)))
        plt.plot(bins[:-1] + binwidth / 2, n.astype(np.float) / np.sum(n),
                 'k', lw=2, drawstyle='steps-mid',
                 label='ridge-following unfiltered')
        plt.xlim([-180, 180])
        plt.ylim([0, 0.12])
        plt.xlabel('Delta Alpha [degrees]')
        plt.ylabel('fraction of tracks per {} degrees'.format(binwidth))
        plt.title('Filter moments by phi and edge_pixel_segments and ' +
                  'edge_pixel_counts')
        plt.legend()
        plt.show()

        # filter with ... and end energy
        lg1 = (np.abs(phi1) < 1.5)
        lg2 = (
            (np.abs(phi2) < 1.5) &
            (edge_pixel_segments2 == 1) &
            (edge_pixel_count2 <= 4) &
            (end_energy2 <= 25))
        lg3 = (end_energy2 <= 25)

        r = np.sum(lg2) / float(len(MR3_da))
        print('moments v3 (phi < 1.5 and edge_pixel_segments = 1 and ' +
              'edge_pixel_count <= 4 and end_energy <= 25) rate: {}'.format(r))
        r = np.sum(lg3) / float(len(HT_da))
        print('ridge-following end_energy <= 25 rate: {}'.format(r))

    if True:
        plt.figure()
        n, bins = np.histogram(
            MR1_da[lg1], np.arange(-180, 180.1, binwidth))
        print('sum(n) = {}'.format(np.sum(n)))
        plt.plot(bins[:-1] + binwidth / 2, n.astype(np.float) / np.sum(n),
                 'b', lw=2, drawstyle='steps-mid',
                 label='moments-1 filtered')
        n, bins = np.histogram(
            MR3_da[lg2], np.arange(-180, 180.1, binwidth))
        print('sum(n) = {}'.format(np.sum(n)))
        plt.plot(bins[:-1] + binwidth / 2, n.astype(np.float) / np.sum(n),
                 'g', lw=2, drawstyle='steps-mid',
                 label='moments-3 filtered')
        n, bins = np.histogram(HT_da[lg3], np.arange(-180, 180.1, binwidth))
        print('sum(n) = {}'.format(np.sum(n)))
        plt.plot(bins[:-1] + binwidth / 2, n.astype(np.float) / np.sum(n),
                 'k', lw=2, drawstyle='steps-mid',
                 label='ridge-following filtered')
        plt.xlim([-180, 180])
        plt.ylim([0, 0.12])
        plt.xlabel('Delta Alpha [degrees]')
        plt.ylabel('fraction of tracks per {} degrees'.format(binwidth))
        plt.title('Filter moments by phi and edge_pixel* and ' +
                  'end_energy')
        plt.legend()
        plt.show()


def eval_by_energy(ARmom):
    """
    take an algresults object, divide up by energy bins, and get parameters
    """
    ARmom.has_beta = False
    bin_edges = np.arange(0.0, 500.0, 50.0)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    ARlist = []

    for i in xrange(len(bin_centers)):
        Emin = bin_edges[i]
        Emax = bin_edges[i + 1]
        ARlist.append(ARmom.select(energy_min=Emin, energy_max=Emax))

    FWHM, FWHM_unc, f, f_unc, f_rej, f_rej_unc = get_uncertainties(ARlist)
    import ipdb; ipdb.set_trace()

    # FWHM
    plt.figure()
    plt.errorbar(bin_centers, FWHM, yerr=FWHM_unc, fmt='o')
    plt.xlim((0, 500))
    plt.ylim((0, 100))
    plt.xlabel('Energy [keV]')
    plt.ylabel('FWHM [degrees]')

    # f
    plt.figure()
    plt.errorbar(bin_centers, f, yerr=f_unc, fmt='o')
    plt.xlim((0, 500))
    plt.ylim((0, 100))
    plt.xlabel('Energy [keV]')
    plt.ylabel('Peak fraction, f [%]')

    # f_rejected
    plt.figure()
    plt.errorbar(bin_centers, f_rej, yerr=f_rej_unc, fmt='o')
    plt.xlim((0, 500))
    plt.ylim((0, 100))
    plt.xlabel('Energy [keV]')
    plt.ylabel('Rejected fraction, f_rej [%]')


def filter_momlist(momlist):
    """
    Filter a moments result list.

    Return momlist_filtered, ends_good, moments_good.
    """

    phi_max = np.pi / 2     # 90 degrees
    edge_segments_max = 1
    edge_pixels_max = 4
    end_energy_max = 25

    phi = np.array([m.phi for m in momlist])
    edge_segments = np.array([m.edge_pixel_segments for m in momlist])
    edge_pixels = np.array([m.edge_pixel_count for m in momlist])
    end_energy = np.array([m.end_energy for m in momlist])

    moments_good = (
        (np.abs(phi) < phi_max) &
        (edge_segments <= edge_segments_max) &
        (edge_pixels <= edge_pixels_max))
    # ends_good applies to hybridtrack also
    ends_good = (end_energy <= end_energy_max)

    all_good = moments_good & ends_good

    momlist_filtered = np.array(momlist)[all_good]

    return momlist_filtered, ends_good, moments_good


def main6(momlist, HTalpha, tracklist):
    """
    Filtered and unfiltered moments and hybridtrack results.
    """

    HTalgname = 'matlab HT v1.5'

    _, ends_good, moments_good = filter_momlist(momlist)

    bin_edges = np.arange(100, 501, 50).astype(float)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # add algorithm outputs to tracklist
    for i in xrange(len(tracklist)):
        if 'moments3' not in tracklist[i].keys():
            add_result(tracklist[i], momlist[i], algname='moments3')
        if 'python HT v1.51c' not in tracklist[i].keys() and HTalpha is not None:
            tracklist[i].add_algorithm('python HT v1.51c', HTalpha[i], np.nan)

    # filter tracklist
    tracklist_ends_good = np.array(tracklist)[ends_good]
    tracklist_moments_good = np.array(tracklist)[ends_good & moments_good]
    print('Tracks (all / ends_good / moments_good): {} / {} / {}'.format(
        len(tracklist), len(tracklist_ends_good), len(tracklist_moments_good)))

    AR = {'HT': {}, 'mom': {}}
    AR['HT']['full'] = ev.AlgorithmResults.from_track_list(
        tracklist, alg_name=HTalgname)
    AR['HT']['good'] = ev.AlgorithmResults.from_track_list(
        tracklist_ends_good, alg_name=HTalgname)
    AR['mom']['full'] = ev.AlgorithmResults.from_track_list(
        tracklist, alg_name='moments3')
    AR['mom']['ends'] = ev.AlgorithmResults.from_track_list(
        tracklist_ends_good, alg_name='moments3')
    AR['mom']['good'] = ev.AlgorithmResults.from_track_list(
        tracklist_moments_good, alg_name='moments3')

    # by energy
    FWHM = {}
    FWHM_unc = {}
    f = {}
    f_unc = {}
    f_fail = {}
    f_fail_unc = {}
    f_rej = {}
    f_rej_unc = {}
    n = {}

    for algkey in AR.iterkeys():
        FWHM[algkey] = {}
        FWHM_unc[algkey] = {}
        f[algkey] = {}
        f_unc[algkey] = {}
        f_fail[algkey] = {}
        f_fail_unc[algkey] = {}
        f_rej[algkey] = {}
        f_rej_unc[algkey] = {}
        n[algkey] = {}

        for filterkey in AR[algkey].iterkeys():
            thisARlist = []
            n[algkey][filterkey] = []
            for i in xrange(len(bin_centers)):
                Emin = bin_edges[i]
                Emax = bin_edges[i + 1]
                cur = AR[algkey][filterkey].select(
                    energy_min=Emin, energy_max=Emax)
                thisARlist.append(cur)
                n[algkey][filterkey].append(len(cur))
            try:
                (FWHM[algkey][filterkey],
                 FWHM_unc[algkey][filterkey],
                 f[algkey][filterkey],
                 f_unc[algkey][filterkey]) = get_uncertainties(thisARlist)
            except ZeroDivisionError:
                print('ZeroDivisionError on {}:{}'.format(algkey, filterkey))
    for algkey in AR.iterkeys():
        for filterkey in AR[algkey].iterkeys():
            f_rej[algkey][filterkey] = np.zeros(len(bin_centers))
            f_rej_unc[algkey][filterkey] = np.zeros(len(bin_centers))
            for i in xrange(len(bin_centers)):
                if filterkey == 'full':
                    f_rej[algkey][filterkey][i] = 0.0
                    f_rej_unc[algkey][filterkey][i] = 0.0
                else:
                    n_tot = float(n[algkey]['full'][i])
                    n_cur = float(n[algkey][filterkey][i])
                    f_rej[algkey][filterkey][i] = (n_tot - n_cur) / n_tot * 100
                    f_rej_unc[algkey][filterkey][i] = (
                        np.sqrt(n_tot - n_cur) / n_tot * 100)


    # import ipdb; ipdb.set_trace()
    lw = 2
    ms = 8

    # plot FWHM
    if True:
        plt.figure()

        algkey, filterkey, mkr, mec, label = (
            'HT', 'full', 'o', 'm', 'Ridge-follow [all]')
        plt.errorbar(
            bin_centers, FWHM[algkey][filterkey],
            yerr=FWHM_unc[algkey][filterkey],
            fmt=mkr + mec, marker=mkr, mfc=mec, mec=mec, lw=lw, ms=ms, label=label)

        algkey, filterkey, mkr, mec, label = (
            'HT', 'good', '*', 'm', 'Ridge-follow [end<25keV]')
        plt.errorbar(
            bin_centers, FWHM[algkey][filterkey],
            yerr=FWHM_unc[algkey][filterkey],
            fmt=mkr + mec, marker=mkr, mfc=mec, mec=mec, lw=lw, ms=ms, label=label)

        algkey, filterkey, mkr, mec, label = (
            'mom', 'full', 'o', 'b', 'Moments [all]')
        plt.errorbar(
            bin_centers, FWHM[algkey][filterkey],
            yerr=FWHM_unc[algkey][filterkey],
            fmt=mkr + mec, marker=mkr, mfc=mec, mec=mec, lw=lw, ms=ms, label=label)

        algkey, filterkey, mkr, mec, label = (
            'mom', 'ends', '*', 'b', 'Moments [end<25keV]')
        plt.errorbar(
            bin_centers, FWHM[algkey][filterkey],
            yerr=FWHM_unc[algkey][filterkey],
            fmt=mkr + mec, marker=mkr, mfc=mec, mec=mec, lw=lw, ms=ms, label=label)

        algkey, filterkey, mkr, mec, label = (
            'mom', 'good', 's', 'b', 'Moments [added filters]')
        plt.errorbar(
            bin_centers, FWHM[algkey][filterkey],
            yerr=FWHM_unc[algkey][filterkey],
            fmt=mkr + mec, marker=mkr, mfc=mec, mec=mec, lw=lw, ms=ms, label=label)
        plt.xlim((0, 500))
        plt.ylim((0, 100))
        plt.xlabel('Energy [keV]')
        plt.ylabel('FWHM [degrees]')
        plt.legend()

    # plot f
    if True:
        plt.figure()

        algkey, filterkey, mkr, mec, label = (
            'HT', 'full', 'o', 'm', 'Ridge-follow [all]')
        plt.errorbar(
            bin_centers, f[algkey][filterkey],
            yerr=f_unc[algkey][filterkey],
            fmt=mkr + mec, marker=mkr, mfc=mec, mec=mec, lw=lw, ms=ms, label=label)

        algkey, filterkey, mkr, mec, label = (
            'HT', 'good', '*', 'm', 'Ridge-follow [end<25keV]')
        plt.errorbar(
            bin_centers, f[algkey][filterkey],
            yerr=f_unc[algkey][filterkey],
            fmt=mkr + mec, marker=mkr, mfc=mec, mec=mec, lw=lw, ms=ms, label=label)

        algkey, filterkey, mkr, mec, label = (
            'mom', 'full', 'o', 'b', 'Moments [all]')
        plt.errorbar(
            bin_centers, f[algkey][filterkey],
            yerr=f_unc[algkey][filterkey],
            fmt=mkr + mec, marker=mkr, mfc=mec, mec=mec, lw=lw, ms=ms, label=label)

        algkey, filterkey, mkr, mec, label = (
            'mom', 'ends', '*', 'b', 'Moments [end<25keV]')
        plt.errorbar(
            bin_centers, f[algkey][filterkey],
            yerr=f_unc[algkey][filterkey],
            fmt=mkr + mec, marker=mkr, mfc=mec, mec=mec, lw=lw, ms=ms, label=label)

        algkey, filterkey, mkr, mec, label = (
            'mom', 'good', 's', 'b', 'Moments [added filters]')
        plt.errorbar(
            bin_centers, f[algkey][filterkey],
            yerr=f_unc[algkey][filterkey],
            fmt=mkr + mec, marker=mkr, mfc=mec, mec=mec, lw=lw, ms=ms, label=label)
        plt.xlim((0, 500))
        plt.ylim((0, 100))
        plt.xlabel('Energy [keV]')
        plt.ylabel('Peak fraction, f [%]')
        plt.legend(loc='lower right')

    # acceptance fractions
    if True:
        plt.figure()
        algkey, filterkey, mkr, mec, label = (
            'mom', 'full', 'o', 'k', 'All tracks')
        plt.errorbar(
            bin_centers, 100 - f_rej[algkey][filterkey],
            yerr=f_rej_unc[algkey][filterkey],
            fmt=mkr + mec, marker=mkr, mfc=mec, mec=mec, lw=lw, ms=ms, label=label)
        algkey, filterkey, mkr, mec, label = (
            'mom', 'ends', 'o', 'b', 'End energy < 25 keV')
        plt.errorbar(
            bin_centers, 100 - f_rej[algkey][filterkey],
            yerr=f_rej_unc[algkey][filterkey],
            fmt=mkr + mec, marker=mkr, mfc=mec, mec=mec, lw=lw, ms=ms, label=label)
        algkey, filterkey, mkr, mec, label = (
            'mom', 'good', 'o', 'g', 'All filters applied')
        plt.errorbar(
            bin_centers, 100 - f_rej[algkey][filterkey],
            yerr=f_rej_unc[algkey][filterkey],
            fmt=mkr + mec, marker=mkr, mfc=mec, mec=mec, lw=lw, ms=ms, label=label)
        plt.xlim((0, 500))
        plt.ylim((0, 100))
        plt.xlabel('Energy [keV]')
        plt.ylabel('Filter acceptance fraction [%]')
        plt.legend(loc='lower right')


def get_uncertainties(ARlist):
    """
    Get FWHM, FWHM_unc, f, f_unc, f_rejected, f_rejected_unc (%)
    """

    FWHM = np.zeros(len(ARlist))
    FWHM_unc = np.zeros(len(ARlist))
    f = np.zeros(len(ARlist))
    f_unc = np.zeros(len(ARlist))

    for i in xrange(len(ARlist)):
        ARlist[i].has_beta = False
        ARlist[i].add_default_uncertainties()
        FWHM[i] = ARlist[i].alpha_unc.metrics['FWHM'].value
        FWHM_unc[i] = ARlist[i].alpha_unc.metrics['FWHM'].uncertainty[0]
        f[i] = ARlist[i].alpha_unc.metrics['f'].value
        f_unc[i] = ARlist[i].alpha_unc.metrics['f'].uncertainty[0]

    return FWHM, FWHM_unc, f, f_unc


def algresults_from_lists(tracklist, momlist, algname='moments'):
    """
    Create an AlgorithmResults instance from tracklist and momlist.
    """
    for i in xrange(len(tracklist)):
        if algname in tracklist[i].algorithms:
            del(tracklist[i].algorithms[algname])
        add_result(tracklist[i], momlist[i], algname=algname)

    algresults = ev.AlgorithmResults.from_track_list(
        tracklist, alg_name=algname)

    return algresults


def add_result(track, mom, algname='moments'):
    """
    Add a moments result to a Track object using Track.add_algorithm().
    """
    # if da is None:
    #     da = mom.alpha * 180 / np.pi - track.g4track.alpha_deg
    # while da > 180:
    #     da -= 360
    # while da < -180:
    #     da += 360
    # db = np.nan     # no beta measurement

    track.add_algorithm(algname, mom.alpha * 180.0 / np.pi, np.nan)


def main7():
    """
    Start from scratch and build a test dataset.
    """

    tracks_nofilter = get_tracklist(n_files=8)  # 8 files: 6705 tracks
    tracks_300 = get_tracklist(n_files=8)     # 8 files: 1786 tracks >300keV

    mom_nofilter = momentlist_from_tracklist(tracks_nofilter)
    mom_300 = momentlist_from_tracklist(tracks_300)

    clist_nofilter = classifierlist_from_tracklist(
        tracks_nofilter, mom_nofilter)
    clist_300 = classifierlist_from_tracklist(tracks_300, mom_300)

    return (tracks_nofilter, tracks_300,
            mom_nofilter, mom_300,
            clist_nofilter, clist_300)

if __name__ == '__main__':
    main2()
