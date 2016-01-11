#!/usr/bin/python

import numpy as np
import os
import glob
import ipdb as pdb
import h5py
import progressbar
import time

import evaluation
import trackdata
import trackio


def main():

    loadpath = '/media/plimley/TEAM 7B/HTbatch01_pyml'
    loadfmt = 'MultiAngle_HT_*_py.h5'

    os.chdir(loadpath)
    flist = glob.glob(loadfmt)

    pixlist = ['10_5', '2_5']
    alglist = ['matlab HT v1.5', 'python HT v1.5', 'python HT v1.5a']

    # results_by_file = {}
    # all_results = {}
    subgroup_name = {}
    save_group_name = {}
    for pix in pixlist:
        # results_by_file[pix] = {}
        # all_results[pix] = {}
        subgroup_name[pix] = 'pix' + pix + 'noise0'
        save_group_name[pix] = {}
        for alg in alglist:
            # results_by_file[pix][alg] = []      # to be appended to
            save_group_name[pix][alg] = alg + ', ' + pix
            # all_results[pix][alg] = []        # to be replaced with an AR obj
    res = []

    pbar = progressbar.ProgressBar(
        widgets=[progressbar.Percentage(), ' ', progressbar.Bar(), ' ',
                 progressbar.ETA()], maxval=len(flist))
    pbar.start()

    for i, f in enumerate(flist):
        savename = make_savename(f)
        if os.path.exists(savename):
            continue
        with h5py.File(f, 'r') as h5load, h5py.File(savename, 'w') as h5save:
            for pix in pixlist:
                for alg in alglist:
                    # time.sleep(1)
                    print 'Reading ' + f
                    res = evaluation.AlgorithmResults.from_hdf5_tracks(
                        h5load, subgroup_name=subgroup_name[pix], alg_name=alg)
                    print '  Writing ' + savename
                    trackio.write_object_to_hdf5(
                        res, h5save, save_group_name[pix][alg])
        pbar.update(i)
    pbar.finish()

    # for pix in pixlist:
    #     for alg in alglist:
    #         all_results[pix][alg] = results_sum(results_by_file[pix][alg])
    #
    # pass


def make_savename(loadname):
    base, _ = os.path.splitext(loadname)
    savename = base + '_AR.h5'
    return savename


def results_sum(results_list):
    """
    Use the python built-in sum function to combine a list of AlgorithmResults
    objects.
    """

    if not results_list:
        return []
    else:
        # must give it the starting value of an AR object
        return sum(results_list[1:], results_list[0])


if __name__ == '__main__':

    cwd = os.getcwd()
    try:
        main()
    finally:
        os.chdir(cwd)

    if False:
        pdb.set_trace()
        pass
