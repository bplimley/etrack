# -*- coding: utf-8 -*-

from __future__ import print_function
import ipdb as pdb
import numpy as np
import h5py
import time
import glob
import os
import multiprocessing
import psutil
import progressbar

from etrack.reconstruction.trackdata import Track, G4Track
import etrack.io.trackio as trackio
import etrack.reconstruction.trackmoments as tm
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

    print('Found {} files to load'.format(len(flist)))

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
    server_flag = False
    if server_flag:
        n_threads = 11
        loadpath = '/global/home/users/bcplimley/multi_angle/DTbatch01_h5'
        savepath = '/global/home/users/bcplimley/multi_angle/algs_10.5_batch01'
    else:
        # LBL desktop
        n_threads = 4
        loadpath = '/media/plimley/TEAM 7B/DTbatch01_h5'
        savepath = '/media/plimley/TEAM 7B/algs_10.5_batch01'
    loadglob = 'MultiAngle_DT_*_*.h5'
    saveglob = 'MultiAngle_algs_*_*.h5'

    v = 3   # verbosity

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

    progressflag = not file_vars()[0]

    pn = 'pix10_5noise15'
    HTname = 'python HT v1.52'
    MTname = 'moments v1.0'

    vprint(v, 1, 'Starting {} at {} with {}% mem usage'.format(
        loadfile, time.ctime(), psutil.virtual_memory().percent))

    pyobj_to_h5 = {}

    try:
        with h5py.File(loadfile, 'a', driver='core') as f, h5py.File(
                savefile, 'a', driver='core') as h5save:
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
                vprint(v, 3, 'Beginning track {} in {}'.format(ind, loadfile))
                if int(ind) % 50 == 0:
                    vprint(v, 2,
                           'Beginning track {} in {}'.format(ind, loadfile))

                # evt is for the G4Track object
                evt = f[ind]
                # shouldn't get a KeyError because we are looping over ind

                # trk is for the Track object
                try:
                    trk = evt[pn]
                except KeyError:
                    vprint(v, 1,
                           '**Missing key {} in {}{}, skipping'.format(
                               pn, loadfile, evt.name))
                    continue

                trkpath = ind + '/' + pn
                n += 1
                if n > 50:
                    # testing
                    # vprint(v, 1, 'Finished 50 files, exiting')
                    # break
                    pass

                # load G4Track
                vprint(v, 3, 'Loading G4Track {} in {}'.format(ind, loadfile))
                try:
                    this_g4track = G4Track.from_dth5(evt)
                except trackio.InterfaceError:
                    vprint(v, 2, 'InterfaceError in G4Track at {}, {}'.format(
                        loadfile, ind))
                    continue

                # load Track
                vprint(v, 3, 'Loading Track {} in {}'.format(ind, loadfile))
                try:
                    this_track = Track.from_dth5(trk, g4track=this_g4track)
                except trackio.InterfaceError:
                    vprint(v, 2, 'InterfaceError in Track at {}, {}'.format(
                        loadfile, ind))
                    continue

                # run moments algorithm
                vprint(v, 3, 'Running moments on track {} in {}'.format(
                    ind, loadfile))
                try:
                    mom = tm.MomentsReconstruction(this_track.image)
                    mom.reconstruct()
                except NotImplementedError:
                    pass
                # any real exceptions?
                # write into track object
                if mom.alpha:
                    this_track.add_algorithm(
                        MTname,
                        alpha_deg=mom.alpha * 180 / np.pi,
                        beta_deg=np.nan, info=None)
                else:
                    this_track.add_algorithm(
                        MTname,
                        alpha_deg=np.nan,
                        beta_deg=np.nan, info=None)
                # write into savefile
                trackio.write_object_to_hdf5(
                    mom, h5save, 'mom_' + trkpath, pyobj_to_h5=pyobj_to_h5)

                # run HT algorithm (v1.52)
                vprint(v, 3, 'Running HT on track {} in {}'.format(
                    ind, loadfile))
                try:
                    _, HTinfo = ht.reconstruct(this_track)
                except (ht.InfiniteLoop, ht.NoEndsFound):
                    # write empty into track object
                    this_track.add_algorithm(
                        HTname,
                        alpha_deg=np.nan,
                        beta_deg=np.nan,
                        info=None)
                else:
                    # trim memory usage
                    if hasattr(HTinfo, 'ridge'):
                        if HTinfo.ridge:
                            for ridgept in HTinfo.ridge:
                                ridgept.cuts = None
                                ridgept.best_cut = None
                    # write into track object
                    this_track.add_algorithm(
                        HTname,
                        alpha_deg=HTinfo.alpha_deg,
                        beta_deg=HTinfo.beta_deg,
                        info=HTinfo)

                # write py HDF5 format into savefile (including alg outputs)
                try:
                    trackio.write_object_to_hdf5(
                        this_track, h5save, trkpath,
                        pyobj_to_h5=pyobj_to_h5)
                except trackio.InterfaceError:
                    vprint(v, 1, 'InterfaceError writing to file at ' +
                           '{}, {}'.format(savefile, ind))
                    continue

                # run classifier
                vprint(v, 3, 'Running MC classifier on track {} in {}'.format(
                    ind, loadfile))
                classifier = cl.Classifier(this_track.g4track)
                try:
                    classifier.mc_classify()
                except cl.TrackTooShortError:
                    classifier.error = 'TrackTooShortError'
                else:
                    vprint(v, 3,
                           'Running ends classifier on track {} in {}'.format(
                               ind, loadfile))
                    try:
                        classifier.end_classify(this_track, mom=mom)
                    except tp.G4TrackTooBigError:
                        classifier.error = 'G4TrackTooBigError'
                # write into savefile
                vprint(v, 3, 'Writing classifier into {} for track {}'.format(
                    savefile, ind))
                trackio.write_object_to_hdf5(
                    classifier, h5save, 'cl_' + trkpath,
                    pyobj_to_h5=pyobj_to_h5)

                if progressflag:
                    pbar.update(n)
            if progressflag:
                pbar.finish()
        # f gets closed here
    except NotImplementedError:
        pass


def test_read():
    """
    Read classifier and moments objects from file, to test data integrity.
    """

    loadpath = '/media/plimley/TEAM 7B/MTbatch01'
    loadname = 'MultiAngle_MT_100_1.h5'
    loadfile = os.path.join(loadpath, loadname)

    clist = trackio.read_object_list_from_hdf5(
        loadfile, cl.Classifier.from_hdf5, prefix='cl_')

    momlist = trackio.read_object_list_from_hdf5(
        loadfile, tm.MomentsReconstruction.from_hdf5, prefix='mom_')

    return clist, momlist


if __name__ == '__main__':
    run_main()

    if False:
        pdb.set_trace()
