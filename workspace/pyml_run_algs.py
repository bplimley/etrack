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
import progressbar

import trackdata
import trackio
import evaluation
import hybridtrack as ht
import hybridtrack2 as ht2
import hybridtrack2c as ht2c
from filejob import JobOptions, vprint


def run_main():

    multi_flag = True   # run in parallel - turn off to debug
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

    server_flag = True
    if server_flag:
        n_threads = 12
        loadpath = '/global/home/users/bcplimley/multi_angle/HTbatch01_pyml'
        savepath = '/global/home/users/bcplimley/multi_angle/HTbatch01_ARnew'
    else:
        # LBL desktop
        n_threads = 4
        loadpath = '/media/plimley/TEAM 7B/HTbatch01_pyml'
        savepath = '/media/plimley/TEAM 7B/HTbatch01_AR'
    loadglob = 'MultiAngle_HT_*_*_py.h5'
    saveglob = 'MultiAngle_HT_*_*_AR.h5'

    v = 1   # verbosity

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
    doneflag = True

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

    progressflag = False     # turn off for parallel

    pnlist = ['pix10_5noise0',
              'pix2_5noise0',
              'pix5noise0',
              'pix20noise0',
              'pix40noise0']
    alglist = {'python HT v1.5': ht,
               'python HT v1.5a': ht2,
               'python HT v1.5c': ht2c}
    tracklist = {}
    AR = {}
    for pnname in pnlist:
        tracklist[pnname] = []
        AR[pnname] = {}
        for algname in alglist.keys():
            AR[pnname][algname] = []

    vprint(v, 1, '\nStarting {} at {}'.format(loadfile, time.ctime()))
    try:
        with h5py.File(loadfile, 'a', driver='core') as h5load:
            filename = h5load.filename
            n = 0
            if progressflag:
                pbar = progressbar.ProgressBar(
                    widgets=[progressbar.Percentage(), ' ',
                             progressbar.Bar(), ' ',
                             progressbar.ETA()], maxval=len(h5load))
                pbar.start()

            keylist = h5load.keys()
            keylist.sort()
            for ind in keylist:
                if int(ind) % 10 == 0:
                    vprint(v, 2, '    Running {} #{} at {}'.format(
                        loadfile, ind, time.ctime()))
                trk = h5load[ind]
                h5_to_pydict = {}
                pydict_to_pyobj = {}
                pyobj_to_h5 = {}

                n += 1
                if n > 50:
                    # pdb.set_trace()
                    # continue  # TODO temp!
                    pass

                for pnname in pnlist:
                    try:
                        pn = trk[pnname]
                    except KeyError:
                        vprint(v, 1,
                               '**Missing key {} in {}{}, skipping'.format(
                                   pnname, loadfile, trk.name))
                        continue
                    vprint(v, 3, 'Running {}{} at {}'.format(
                        loadfile, pn.name, time.ctime()))
                    # load track
                    try:
                        this_track = trackdata.Track.from_hdf5(
                            pn,
                            h5_to_pydict=h5_to_pydict,
                            pydict_to_pyobj=pydict_to_pyobj)
                    except trackio.InterfaceError:
                        vprint(v, 3, 'InterfaceError at {}{}'.format(
                            loadfile, pn.name))
                        continue
                    tracklist[pnname].append(this_track)
                    # each algorithm version
                    for algname, algfunc in alglist.items():
                        vprint(v, 4, 'Running alg {} at {}'.format(
                            algname, time.ctime()))
                        # check for result
                        if algname not in this_track.algorithms:
                            # run algorithm
                            try:
                                _, HTinfo = algfunc.reconstruct(this_track)
                            except algfunc.InfiniteLoop:
                                continue
                            except algfunc.NoEndsFound:
                                continue
                            # trim memory usage!
                            if hasattr(HTinfo, 'ridge'):
                                if HTinfo.ridge:
                                    for ridgept in HTinfo.ridge:
                                        ridgept.cuts = None
                                        ridgept.best_cut = None
                            # write into track object
                            try:
                                this_track.add_algorithm(
                                    algname,
                                    alpha_deg=HTinfo.alpha_deg,
                                    beta_deg=HTinfo.beta_deg,
                                    info=HTinfo)
                                # write into HDF5
                                trackio.write_object_to_hdf5(
                                    this_track.algorithms[algname],
                                    pn['algorithms'], algname,
                                    pyobj_to_h5=pyobj_to_h5)
                            except trackdata.InputError:
                                # already has this algorithm
                                pass

                if progressflag:
                    pbar.update(n)
            if progressflag:
                pbar.finish()
            # h5load gets closed
            vprint(v, 2, '\n  Finished loading {} at {}'.format(
                loadfile, time.ctime()))
    except IOError:
        vprint(v, 1, 'IOError: Unable to open file (I think) for {}'.format(
            loadfile))
        return None

    # AlgorithmResults objects
    alglist2 = alglist.keys() + ['matlab HT v1.5']
    for pnname in pnlist:
        for algname in alglist2:
            this_AR = evaluation.AlgorithmResults.from_track_array(
                tracklist[pnname], alg_name=algname, filename=filename)
            AR[pnname][algname] = this_AR
    vprint(v, 2, '\n  Created AR objects for {} at {}'.format(
        loadfile, time.ctime()))

    # write to savefile
    try:
        with h5py.File(savefile, 'w', driver='core') as h5save:
            for pnname, AR_pn in AR.items():
                pngroup = h5save.create_group(pnname)
                for algname, this_AR in AR_pn.items():
                    trackio.write_object_to_hdf5(this_AR, pngroup, algname)
    except IOError:
        vprint(v, 1, 'IOError: Unable to create file (I think) for {}'.format(
            savefile))

    return None


if __name__ == '__main__':
    run_main()

    if False:
        pdb.set_trace()
