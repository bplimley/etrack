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
import socket

from etrack.reconstruction.trackdata import Track
import etrack.io.trackio as trackio
import etrack.reconstruction.trackmoments as tm
import etrack.reconstruction.classify as cl
from etrack.workspace.filejob import JobOptions, vprint

# runs after classify_all.py
# go through all the MultiAngle_algs_*.h5,
#   collects the results numbers, and saves into a results file.


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
    if socket.gethostname() == 'plimley-Vostro-mint17':
        server_flag = False
    elif socket.gethostname().startswith('n0'):
        server_flag = True

    if server_flag:
        n_threads = 11
        loadpath = '/global/home/users/bcplimley/multi_angle/algs_10.5_batch01'
        savepath = '/global/home/users/bcplimley/multi_angle/clresults_10.5_batch01'
    else:
        # LBL desktop
        n_threads = 4
        loadpath = '/media/plimley/TEAM 7B/algs_10.5_batch01'
        savepath = '/media/plimley/TEAM 7B/clresults_10.5_batch01'
    loadglob = 'MultiAngle_algs_*_*.h5'
    saveglob = 'MultiAngle_results_*_*.h5'

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
    phflag = False
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
        get_results(loadfile, savefile, v)
        # clean up
        opts.post_job_tasks(loadname)


def get_results(loadfile, savefile, v):
    """
    Get all the important numbers for each event.

    (basic info:)
        Etot
        Edep
        Etrack
        alpha_true
        beta_true
        errorcode
        file
        ind
    (ridge following result:)
        alpha_rf
    (moments result:)
        alpha_m
    (general rejection parameters:)
        min_end_energy
        max_end_energy
        n_ends
    (moments rejection parameters:)
        phi
        edge_pixels
        edge_segments
    (Monte Carlo rejection parameters:)
        overlap flag
        wrong_end flag
        early_scatter flag
        total_scatter_angle
    """

    progressflag = not file_vars()[0]

    pn = 'pix10_5noise15'
    HTname = 'python HT v1.52'
    MTname = 'moments v1.0'

    vprint(v, 1, 'Starting {} at {} with {}% mem usage'.format(
        loadfile, time.ctime(), psutil.virtual_memory().percent))

    datalen = 1000
    # basic info
    energy_tot_kev = np.nan * np.ones(datalen)
    energy_dep_kev = np.nan * np.ones(datalen)
    energy_track_kev = np.nan * np.ones(datalen)
    alpha_true_deg = np.nan * np.ones(datalen)
    beta_true_deg = np.nan * np.ones(datalen)
    trk_errorcode = np.nan * np.ones(datalen)
    cl_errorcode = np.nan * np.ones(datalen)
    mom_errorcode = np.nan * np.ones(datalen)
    filename = ['' for _ in xrange(datalen)]
    fileind = np.nan * np.ones(datalen)
    # ridge following
    alpha_ridge_deg = np.nan * np.ones(datalen)
    # moments
    alpha_moments_deg = np.nan * np.ones(datalen)
    # rejection
    min_end_energy_kev = np.nan * np.ones(datalen)
    max_end_energy_kev = np.nan * np.ones(datalen)
    n_ends = np.nan * np.ones(datalen)
    phi_deg = np.nan * np.ones(datalen)
    edge_pixels = np.nan * np.ones(datalen)
    edge_segments = np.nan * np.ones(datalen)
    # monte carlo
    overlap_flag = np.nan * np.ones(datalen)
    wrong_end_flag = np.nan * np.ones(datalen)
    early_scatter_flag = np.nan * np.ones(datalen)
    total_scatter_angle_deg = np.nan * np.ones(datalen)

    try:
        with h5py.File(loadfile, 'r', driver='core') as f, h5py.File(
                savefile, 'a', driver='core') as h5save:
            if progressflag:
                pbar = progressbar.ProgressBar(
                    widgets=[progressbar.Percentage(), ' ',
                             progressbar.Bar(), ' ',
                             progressbar.ETA()], maxval=datalen)
                pbar.start()

            for ind in xrange(datalen):
                indstr = '{:05d}'.format(ind)
                trkstr = indstr
                clstr = 'cl_' + indstr
                momstr = 'mom_' + indstr

                this_trk_errorcode = 0
                this_cl_errorcode = 0
                this_mom_errorcode = 0

                filename[ind] = os.path.split(loadfile)[-1]
                fileind[ind] = ind

                try:
                    this_trk = Track.from_hdf5(f[trkstr][pn])
                # except KeyError:
                #     pass
                except trackio.InterfaceError:
                    read_errorcode = f[trkstr][pn].attrs['errorcode']
                    if read_errorcode > 0:
                        # multiplicity event
                        this_trk_errorcode = read_errorcode
                        continue
                energy_tot_kev[ind] = this_trk.g4track.energy_tot_kev
                energy_dep_kev[ind] = this_trk.g4track.energy_dep_kev
                energy_track_kev[ind] = this_trk.energy_kev
                alpha_true_deg[ind] = this_trk.g4track.alpha_deg
                beta_true_deg[ind] = this_trk.g4track.beta_deg
                alpha_ridge_deg[ind] = this_trk[HTname].alpha_deg
                alpha_moments_deg[ind] = this_trk[MTname].alpha_deg

                # try:
                this_cl = cl.Classifier.from_hdf5(f[clstr][pn])
                # except KeyError:
                #     pass
                min_end_energy_kev[ind] = this_cl.min_end_energy
                max_end_energy_kev[ind] = this_cl.max_end_energy
                n_ends[ind] = this_cl.n_ends
                overlap_flag[ind] = this_cl.overlap
                wrong_end_flag[ind] = this_cl.wrong_end
                if this_cl.error == 'TrackTooShortError':
                    this_cl_errorcode = 8
                else:
                    early_scatter_flag[ind] = this_cl.early_scatter
                    total_scatter_angle_deg[ind] = (
                        this_cl.total_scatter_angle / np.pi * 180)

                try:
                    this_mom = tm.MomentsReconstruction.from_hdf5(
                        f[momstr][pn])
                except trackio.InterfaceError:
                    if f[momstr][pn].attrs['errorcode'] == 4:
                        # no ends found
                        this_mom_errorcode = 4
                    else:
                        raise
                else:
                    if this_mom.error == 'CheckSegmentBoxError':
                        this_mom_errorcode = 9
                    else:
                        phi_deg[ind] = this_mom.phi / np.pi * 180
                        edge_pixels[ind] = this_mom.edge_pixel_count
                        edge_segments[ind] = this_mom.edge_pixel_segments

                trk_errorcode[ind] = this_trk_errorcode
                cl_errorcode[ind] = this_cl_errorcode
                mom_errorcode[ind] = this_mom_errorcode

                if progressflag:
                    pbar.update(ind)

            # save to file
            # basic info
            h5save.create_dataset(
                'energy_tot_kev', shape=(datalen,), data=energy_tot_kev)
            h5save.create_dataset(
                'energy_dep_kev', shape=(datalen,), data=energy_dep_kev)
            h5save.create_dataset(
                'energy_track_kev', shape=(datalen,), data=energy_track_kev)
            h5save.create_dataset(
                'alpha_true_deg', shape=(datalen,), data=alpha_true_deg)
            h5save.create_dataset(
                'beta_true_deg', shape=(datalen,), data=beta_true_deg)
            h5save.create_dataset(
                'trk_errorcode', shape=(datalen,), data=trk_errorcode,
                dtype='f2')
            h5save.create_dataset(
                'cl_errorcode', shape=(datalen,), data=cl_errorcode,
                dtype='f2')
            h5save.create_dataset(
                'mom_errorcode', shape=(datalen,), data=mom_errorcode,
                dtype='f2')
            h5save.create_dataset(
                'filename', shape=(datalen,), data=filename,
                dtype=h5py.special_dtype(vlen=str))
            h5save.create_dataset(
                'fileind', shape=(datalen,), data=fileind,
                dtype='f4')
            # ridge following
            h5save.create_dataset(
                'alpha_ridge_deg', shape=(datalen,), data=alpha_ridge_deg)
            # moments
            h5save.create_dataset(
                'alpha_moments_deg', shape=(datalen,), data=alpha_moments_deg)
            # rejection parameters
            h5save.create_dataset(
                'min_end_energy_kev', shape=(datalen,),
                data=min_end_energy_kev)
            h5save.create_dataset(
                'max_end_energy_kev', shape=(datalen,),
                data=max_end_energy_kev)
            h5save.create_dataset(
                'n_ends', shape=(datalen,), data=n_ends,
                dtype='f4')
            h5save.create_dataset(
                'phi_deg', shape=(datalen,), data=phi_deg)
            h5save.create_dataset(
                'edge_pixels', shape=(datalen,), data=edge_pixels,
                dtype='f4')
            h5save.create_dataset(
                'edge_segments', shape=(datalen,), data=edge_segments,
                dtype='f2')
            # Monte Carlo
            h5save.create_dataset(
                'overlap_flag', shape=(datalen,), data=overlap_flag,
                dtype='f2')
            h5save.create_dataset(
                'wrong_end_flag', shape=(datalen,), data=wrong_end_flag,
                dtype='f2')
            h5save.create_dataset(
                'early_scatter_flag', shape=(datalen,),
                data=early_scatter_flag, dtype='f2')
            h5save.create_dataset(
                'total_scatter_angle_deg', shape=(datalen,),
                data=total_scatter_angle_deg)

            if progressflag:
                pbar.finish()

    except NotImplementedError:
        pass


if __name__ == '__main__':
    run_main()

    if False:
        pdb.set_trace()
