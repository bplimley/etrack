# -*- coding: utf-8 -*-

from __future__ import print_function
import ipdb as pdb
import numpy as np
import h5py
import time
import datetime
import glob
import os
import multiprocessing
from datetime import datetime as dt
import psutil
import progressbar

from etrack.reconstruction.trackdata import Track
import etrack.io.trackio as trackio
import etrack.reconstruction.trackmoments as tm
import etrack.reconstruction.evaluation as ev
import etrack.reconstruction.classify as cl
import etrack.reconstruction.hybridtrack as ht
import etrack.visualization.trackplot as tp
from etrack.workspace.filejob import JobOptions, vprint


# copied from filejob.py
def run_main():

    multi_flag, _, loadpath, savepath, loadglob, saveglob, v, n_proc = (
        file_vars())

    if not os.path.isdir(savepath):
        os.mkdir(savepath)

    flist = glob.glob(os.path.join(loadpath, loadglob))
    flist.sort()

    if multi_flag:
        p = multiprocessing.Pool(processes=n_proc, maxtasksperchild=5)
        p.map(runfile, flist, chunksize=5)
    else:
        [runfile(f) for f in flist]


def file_vars():
    """
    Define file path and globs here. Also server flag, and verbosity.

    Gets loaded in run_main() as well as runfile(loadname).
    """

    multi_flag = False   # run in parallel - turn off to debug
    server_flag = True
    if server_flag:
        n_threads = 11
        loadpath = '/global/home/users/bcplimley/multi_angle/HTbatch01_pyml'
        savepath = '/global/home/users/bcplimley/multi_angle/MTbatch01'
    else:
        # LBL desktop
        n_threads = 4
        loadpath = '/media/plimley/TEAM 7B/HTbatch01_pyml'
        savepath = '/media/plimley/TEAM 7B/MTbatch01'
    loadglob = 'MultiAngle_HT_*_*_py.h5'
    saveglob = 'MultiAngle_MT_*_*.h5'

    v = 1   # verbosity

    return (multi_flag, server_flag, loadpath, savepath, loadglob, saveglob, v,
            n_threads)


def runfile(loadname):
    """
    To use:
      1. make a copy in your script
      2. edit this function name to e.g. runmyjob(loadname)
      3. edit work function name from main_work_function below
      4. edit paths, globs, and flags
      5. in the main script:
         flist = glob.glob(os.path.join(loadpath, loadglob))
         p = multiprocessing.Pool(processes=n, maxtasksperchild=25)
         p.map(runmyjob, flist, chunksize=25)
    """

    # drop the path part of loadname, if it is given
    multi_flag, server_flag, loadpath, savepath, loadglob, saveglob, v, _ = (
        file_vars())

    in_place_flag = False
    phflag = True
    doneflag = False

    # setup
    opts = JobOptions(
        loadpath=loadpath, loadglob=loadglob,
        savepath=savepath, saveglob=saveglob,
        in_place_flag=in_place_flag, phflag=phflag, doneflag=doneflag)
    # decide to skip or not; construct full filenames
    loadfile, savefile = opts.pre_job_tasks(loadname)
    if loadfile is not None and savefile is not None:
        # do the work
        classify_etc(loadfile, savefile, v)
        # clean up
        opts.post_job_tasks(loadname)


def classify_etc(loadfile, savefile, v):
    """
    1. Moments algorithm
    2. HybridTrack algorithm
    3. Classify
    """

    multi_flag, *args = file_vars()
    progressflag = not multi_flag

    pn = 'pix10_5noise15'

    vprint(v, 1, 'Starting {} at {} with {}% mem usage'.format(
        loadfile, time.ctime(), psutil.virtual_memory().percent))

    try:
        with h5py.File(loadfile, 'a', driver='core') as f:
            #
            n = 0
            if progressflag:
                pbar = progressbar.ProgressBar(
                    widgets=[progressbar.Percentage(), ' ',
                             progressbar.Bar(), ' ',
                             progressbar.ETA()], maxval=len(f))
                pbar.start()

            keylist = f.keys()
            keylist.sort()
            for ind in keylist:
                # if int(ind) % 10 == 0:
                #     ...
                try:
                    trk = f[ind][pn]
                except KeyError:
                    vprint(v, 1,
                           '**Missing key {} in {}{}, skipping'.format(
                               pnname, loadfile, trk.name))
                    continue
                h5_to_pydict = {}
                pydict_to_pyobj = {}
                pyobj_to_h5 = {}

                n += 1
                if n > 50:
                    # testing
                    break

                # ...

                if progressflag:
                    pbar.update(n)
            if progressflag:
                pbar.finish()
        # f gets closed


if __name__ == '__main__':
    run_main()

    if False:
        pdb.set_trace()
