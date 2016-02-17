# pyml_run_algs.py
#
# comes after h5m_to_pyml.py
#
# 1. tracks from pyml folder, pix10_5noise0, pix2_5noise0
# 2. run hybridtrack, hybridtrack2, hybridtrack2b on tracks
# 3. save results back into track objects
# 4. also, compile results in AR files in HTbatch01_AR

from __future__ import print_function

import os
import time
import glob
import h5py
import multiprocessing
import ipdb as pdb

import trackdata
import trackio
import evaluation
import hybridtrack as ht
import hybridtrack2 as ht2
import hybridtrack2b as ht2b
from filejob import JobOptions, vprint


def run_main():

    multi_flag = False   # run in parallel - turn off to debug
    _, loadpath, savepath, loadglob, saveglob, v, n_proc = file_vars()

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

    server_flag = False
    if server_flag:
        n_threads = 11
        loadpath = '/global/home/users/bcplimley/multi_angle/HTbatch01_pyml'
        savepath = '/global/home/users/bcplimley/multi_angle/HTbatch01_AR'
    else:
        # LBL desktop
        n_threads = 4
        loadpath = '/media/plimley/TEAM 7B/HTbatch01_pyml'
        savepath = '/media/plimley/TEAM 7B/HTbatch01_AR'
    loadglob = 'MultiAngle_HT_*_*_py.h5'
    saveglob = 'MultiAngle_HT_*_*_AR.h5'

    v = 2   # verbosity

    return server_flag, loadpath, savepath, loadglob, saveglob, v, n_threads


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
    server_flag, loadpath, savepath, loadglob, saveglob, v, _ = file_vars()

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
        pyml_run_algs(loadfile, savefile, v)
        # clean up
        opts.post_job_tasks(loadname)


def pyml_run_algs(loadfile, savefile, v):

    pnlist = ['pix10_5noise0', 'pix2_5noise0']
    alglist = {'python HT v1.5': ht,
               'python HT v1.5a': ht2,
               'python HT v1.5b': ht2b}
    tracklist = {}
    AR = {}
    for pnname in pnlist:
        tracklist[pnname] = []
        AR[pnname] = {}
        for algname in alglist.keys():
            AR[pnname][algname] = []

    vprint(v, 1, 'Starting {} at {}'.format(loadfile, time.ctime()))
    with h5py.File(loadfile, 'a', driver='core') as h5load:
        filename = h5load.filename
        for trk in h5load.values():
            h5_to_pydict = {}
            pydict_to_pyobj = {}
            pyobj_to_h5 = {}

            for pnname in pnlist:
                pn = trk[pnname]
                # load track
                try:
                    this_track = trackdata.Track.from_hdf5(
                        pn,
                        h5_to_pydict=h5_to_pydict,
                        pydict_to_pyobj=pydict_to_pyobj)
                except trackio.InterfaceError:
                    vprint(v, 2, 'InterfaceError at {}{}'.format(
                        loadfile, pn.name))
                    continue
                tracklist[pnname].append(this_track)
                # each algorithm version
                for algname, algfunc in alglist.items():
                    vprint(v, 3, 'Running alg {} at {}'.format(
                        algname, time.ctime()))
                    # run algorithm
                    HToutput, HTinfo = algfunc.reconstruct(this_track)
                    # write into track object
                    this_track.add_algorithm(
                        algname,
                        alpha_deg=HTinfo.alpha_deg, beta_deg=HTinfo.beta_deg,
                        info=HTinfo)
                    # write into HDF5
                    trackio.write_object_to_hdf5(
                        this_track.algorithms[algname],
                        pn['algorithms'], algname,
                        pyobj_to_h5=pyobj_to_h5)
        # h5load gets closed
        vprint(v, 2, '  Finished loading {} at {}'.format(
            loadfile, time.ctime()))

    # AlgorithmResults objects
    for pnname in pnlist:
        for algname in alglist.keys():
            this_AR = evaluation.AlgorithmResults.from_track_array(
                tracklist[pnname], alg_name=algname, filename=filename)
            AR[pnname][algname] = this_AR
    vprint(v, 2, '  Created AR objects for {} at {}'.format(
        loadfile, time.ctime()))

    # write to savefile
    with h5py.File(savefile, 'w', driver='core') as h5save:
        for pnname, AR_pn in AR.items():
            pngroup = h5save.create_group(pnname)
            for algname, this_AR in AR_pn.items():
                trackio.write_object_to_hdf5(this_AR, pngroup, algname)


if __name__ == '__main__':
    run_main()

    if False:
        pdb.set_trace()
