# -*- coding: utf-8 -*-

from __future__ import print_function
import ipdb as pdb
import numpy as np
import h5py
import time
import glob
import os
import progressbar
import socket

"""
Goes through MultiAngle_results_* and assembles all the variables to one file.
"""


def run_main():

    loadpath, savepath, loadglob, savename = file_vars()

    flist = glob.glob(os.path.join(loadpath, loadglob))
    numfiles = len(flist)
    print('Found {} files to aggregate'.format(numfiles))

    # init
    datalen = numfiles * 1000
    varlist = data_variable_list()

    datadict = {}
    for key in varlist:
        datadict[key] = np.empty(shape=(datalen,))
    datadict['filename'] = np.empty(shape=(datalen,), dtype='|S28')

    n = 0
    pbar = progressbar.ProgressBar(
        widgets=[progressbar.Percentage(), ' ',
                 progressbar.Bar(), ' ',
                 progressbar.ETA()], maxval=numfiles)
    pbar.start()
    for i, f in enumerate(flist):
        n = add_file_data(f, datadict, n)
        pbar.update(i)
    pbar.finish()

    save_compiled_data(savepath, savename, datadict)


def file_vars():
    """
    Return loadpath, savepath, loadglob.
    """
    if socket.gethostname() == 'plimley-Vostro-mint17':
        server_flag = False
    elif socket.gethostname().startswith('n0'):
        server_flag = True

    if server_flag:
        loadpath = ('/global/home/users/bcplimley/multi_angle/' +
                    'clresults_10.5_batch01')
        savepath = loadpath
    else:
        loadpath = '/media/plimley/TEAM 7B/clresults_10.5_batch01'
        savepath = loadpath

    loadglob = 'MultiAngle_results_*_*.h5'
    savename = 'compiled_results.h5'

    return loadpath, savepath, loadglob, savename


def data_variable_list():
    """
    Return a list of all the data variables in the files.
    """
    varlist = (
        'energy_tot_kev',
        'energy_dep_kev',
        'energy_track_kev',
        'alpha_true_deg',
        'beta_true_deg',
        'trk_errorcode',
        'cl_errorcode',
        'mom_errorcode',
        'filename',
        'fileind',
        'alpha_ridge_deg',
        'alpha_moments_deg',
        'min_end_energy_kev',
        'max_end_energy_kev',
        'n_ends',
        'phi_deg',
        'edge_pixels',
        'edge_segments',
        'overlap_flag',
        'wrong_end_flag',
        'early_scatter_flag',
        'total_scatter_angle_deg',
    )
    return varlist


def add_file_data(filename, datadict, n):
    """
    Load one file, and put all the data into datadict.
    """

    varlist = data_variable_list()
    with h5py.File(filename, 'r') as h5f:
        datalen = h5f[varlist[0]].shape[0]
        ind1 = n
        ind2 = n + datalen
        # tmp = np.empty(shape=(datalen,))

        for key in varlist:
            this = h5f[key]
            this.read_direct(datadict[key][ind1:ind2])

    n += datalen

    return n


def save_compiled_data(savepath, savename, datadict):
    """
    Save the data into a single compiled results file.
    """

    varlist = data_variable_list()
    with h5py.File(os.path.join(savepath, savename), 'w') as h5f:
        for key in varlist:
            h5f[key] = datadict[key]


if __name__ == '__main__':
    run_main()
