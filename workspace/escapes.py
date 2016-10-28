
import glob
import h5py
import os
import numpy as np
import datetime
import matplotlib.pyplot as plt

import etrack.reconstruction.trackdata as td
import etrack.reconstruction.trackmoments as tm
import etrack.reconstruction.classify as cl
import etrack.io.trackio as trackio
from etrack.workspace.make_bins import bin_centers


def get_objs(h5f, ind):
    """
    Get the G4Track and Classifier objects from an open h5 file. ind is an int
    """

    pn = 'pix10_5noise15'
    indstr = '{:05d}'.format(ind)

    track = td.Track.from_hdf5(h5f[indstr][pn])
    g4track = track.g4track

    classifier = cl.Classifier.from_hdf5(h5f['cl_' + indstr][pn])

    return g4track, classifier


def is_escape(g4track):
    """
    See if a G4Track is an escape event or not. True = escape
    """

    if np.abs(g4track.energy_tot_kev - g4track.energy_dep_kev) > 2.0:
        escaped = True
    else:
        escaped = False

    return escaped


def run_file(fname):
    """
    Get data from one algs_10.5 file
    """

    escaped = np.empty(1000)
    endmax = np.empty(1000)
    Etot = np.empty(1000)

    with h5py.File(fname, 'r') as h5f:
        for ind in xrange(1000):
            try:
                g4t, clsf = get_objs(h5f, ind)
            except trackio.InterfaceError:
                escaped[ind] = np.nan
                endmax[ind] = np.nan
                Etot[ind] = np.nan
                continue
            escaped[ind] = is_escape(g4t)
            endmax[ind] = clsf.max_end_energy
            Etot[ind] = g4t.energy_tot_kev

    return escaped, endmax, Etot


def get_main():
    """
    Run files in '/media/plimley/TEAM 7B/algs_10.5_batch01'
    """

    loadpath = '/media/plimley/TEAM 7B/algs_10.5_batch01'
    loadglob = 'MultiAngle_algs_*.h5'
    flist = glob.glob(os.path.join(loadpath, loadglob))

    datalen = 1000 * len(flist)
    escaped = np.empty(datalen)
    endmax = np.empty(datalen)
    Etot = np.empty(datalen)

    for i, fname in enumerate(flist):
        print('Starting {} at {}'.format(fname, datetime.datetime.now()))
        _esc, _end, _Etot = run_file(fname)
        escaped[i * 1000:(i + 1) * 1000] = _esc
        endmax[i * 1000:(i + 1) * 1000] = _end
        Etot[i * 1000:(i + 1) * 1000] = _Etot

    return escaped, endmax, Etot


def get_sel(Etot, escaped, energy_thresh=100):
    """
    Get selection vectors (boolean) for escapes and non-escapes, with threshold
    """

    sel_esc = (Etot > energy_thresh) & (escaped == 1)
    sel_non = (Etot > energy_thresh) & (escaped == 0)

    return sel_esc, sel_non


def scatter_plot(escaped, endmax, Etot, energy_thresh=100):
    """
    Generate scatter plot of escaped end max energy vs. non-escaped.
    """

    sel_esc, sel_non = get_sel(Etot, escaped, energy_thresh=energy_thresh)

    plt.figure()
    plt.axes()
    plt.plot(Etot[sel_esc], endmax[sel_esc], '.r', label='escapes')
    plt.plot(Etot[sel_non], endmax[sel_non], '.g', label='non escapes')

    plt.xlim((100, 500))
    plt.ylim((0, 200))
    plt.xlabel('Total track energy [keV]')
    plt.ylabel('Maximum end energy [keV]')
    plt.legend()


def hist_plot(escaped, endmax, Etot, energy_thresh=100, decision_thresh=None):
    """
    Generate histograms of escaped and non-escaped events.
    """

    sel_esc, sel_non = get_sel(Etot, escaped, energy_thresh=energy_thresh)

    dx = 5
    bin_edges = np.arange(0, 200, dx)
    bin_cent = bin_centers(bin_edges)
    n_esc, _ = np.histogram(endmax[sel_esc], bins=bin_edges)
    n_non, _ = np.histogram(endmax[sel_non], bins=bin_edges)

    plt.figure()
    plt.axes()
    plt.plot(bin_cent, n_esc / float(np.sum(n_esc)), 'r',
             lw=2, drawstyle='steps-mid', label='escapes')
    plt.plot(bin_cent, n_non / float(np.sum(n_non)), 'g',
             lw=2, drawstyle='steps-mid', label='non escapes')

    plt.xlim((0, 200))
    plt.xlabel('Maximum end energy [keV]')
    plt.ylabel('Fraction of events per {} keV'.format(dx))

    if decision_thresh:
        n_P = np.sum(sel_esc & np.logical_not(np.isnan(endmax)))
        n_N = np.sum(sel_non & np.logical_not(np.isnan(endmax)))
        n_TP = np.sum(sel_esc & (endmax < decision_thresh))
        n_TN = np.sum(sel_non & (endmax > decision_thresh))
        n_FP = np.sum(sel_non & (endmax < decision_thresh))
        n_FN = np.sum(sel_esc & (endmax > decision_thresh))

        assert n_TP + n_FN == n_P
        assert n_TN + n_FP == n_N

        TPR = float(n_TP) / n_P
        TNR = float(n_TN) / n_N
        FPR = float(n_FP) / n_N
        FNR = float(n_FN) / n_P

        print('TPR: {}, FNR: {}'.format(TPR, FNR))
        print('TNR: {}, FPR: {}'.format(TNR, FPR))

        plt.plot(decision_thresh * np.ones(2), [0, 0.15], '--k',
                 lw=2, label='Decision threshold')

    plt.legend()
