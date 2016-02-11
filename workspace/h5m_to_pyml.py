# h5m_to_pyml.py
#
# Adapted from h5matlab_to_py_LRC.py, but using the job_handler.py toolbox.
#
# Go through the HDF5 files written by MATLAB
#   (that is, write_h5matlab.m calling write_hdf5.m)
# and load into python, and save using my native python write_object_to_hdf5().
#
# Load dir: HTbatch01_h5m
# Save dir: HTbatch01_pyml

from __future__ import print_function

import os
import glob
import h5py
import time
import numpy as np
import ipdb as pdb
import multiprocessing

import trackio
import trackdata
import evaluation
import hybridtrack
import filejob


def run_main():

    multi = True
    LRC_flag, loadpath, savepath, loadglob, saveglob, v, n = file_vars()

    flist = glob.glob(os.path.join(loadpath, loadglob))
    flist.sort()

    if multi:
        p = multiprocessing.Pool(processes=n, maxtasksperchild=25)
        p.map(runfile, flist, chunksize=25)
    else:
        [runfile(f) for f in flist]


def file_vars():
    """
    Define file path and globs here. Also LRC flag, and verbosity.

    Gets loaded in run_main() as well as runfile(loadname).
    """
    LRC_flag = True
    if LRC_flag:
        n_threads = 4
        loadpath = '/global/home/users/bcplimley/multi_angle/HTbatch01_h5m/'
        savepath = '/global/home/users/bcplimley/multi_angle/HTbatch01_pyml/'
    else:
        # LBL desktop
        n_threads = 2
        loadpath = '/media/plimley/TEAM 7B/HTbatch01_h5m/'
        savepath = '/media/plimley/TEAM 7B/HTbatch01_pyml/'
    loadglob = 'MultiAngle_HT_*_*.h5'
    saveglob = 'MultiAngle_HT_*_*_py.h5'

    VERBOSITY = 2
    v = VERBOSITY       # for conciseness

    return LRC_flag, loadpath, savepath, loadglob, saveglob, v, n_threads


def runfile(loadname):
    """
    Handle one file, including skipping and placeholder operations.

    For the main work, it runs h5m_to_pyml.
    """

    # drop the path part of the name
    loadname = os.path.split(loadname)[-1]

    # paths, globs, flags
    LRC_flag, loadpath, savepath, loadglob, saveglob, v, _ = file_vars()
    in_place_flag = False
    phflag = True
    doneflag = False

    # setup
    opts = filejob.JobOptions(
        loadpath=loadpath, loadglob=loadglob,
        savepath=savepath, saveglob=saveglob,
        in_place_flag=in_place_flag, phflag=phflag, doneflag=doneflag)
    # decide to skip or not; construct full filenames
    loadfile, savefile = opts.pre_job_tasks(loadname)
    if loadfile is not None and savefile is not None:
        # do the work
        h5m_to_pyml(loadfile, savefile, v)
        # clean up
        opts.post_job_tasks(loadname)


def h5m_to_pyml(loadfile, savefile, v):
    """
    Main work
    """

    filejob.vprint(
        v, 2, '~ Loading ' + loadfile + ' at ' + time.ctime() + ' ~')
    g4tracks, pixnoise = pyobj_from_h5(loadfile, v)
    filejob.vprint(
        v, 1, '> Saving ' + savefile + ' at ' + time.ctime() + ' <')
    pyobjs_to_h5(g4tracks, pixnoise, savefile, v)
    filejob.vprint(
        v, 2, '= Finished ' + savefile + ' at ' + time.ctime() + ' =')


def pyobj_from_h5(h5filename, v):
    """
    Given a file in h5matlab format, return list of objects
      (trackdata.G4Track and trackdata.Track)

    h5filename: complete filename string
    """

    n_to_run = 10000
    n_run = 0

    with h5py.File(h5filename, 'r', driver='core') as f:
        # get max index number
        indices = []
        for key in f.keys():
            indices.append(int(key))
        max_index = np.max(np.array(indices))
        filejob.vprint(v, 3, '\n Got max_index at ' + time.ctime())

        # initialize array of g4tracks, and dict of diffused tracks
        g4tracks = [[] for i in range(max_index + 1)]
        pixnoise = {}

        # construct tracks
        for track_key in f.keys():
            if n_run > n_to_run:
                continue
            filejob.vprint(
                v, 3, 'Starting ' + track_key + ' at ' + time.ctime())
            n_run += 1

            evt = f[track_key]
            ind = int(track_key)
            g4tracks[ind] = trackdata.G4Track.from_h5matlab(evt)
            for pn_key in evt.keys():
                if not pn_key.startswith('pix'):
                    continue
                if pn_key not in pixnoise:
                    # initialize pixnoise dict entry
                    pixnoise[pn_key] = [[] for i in range(max_index + 1)]
                # add to existing dict
                this_track = trackdata.Track.from_h5matlab(
                    evt[pn_key], g4track=g4tracks[ind])
                # debug
                if this_track is None:
                    pdb.set_trace()
                pixnoise[pn_key][ind] = this_track

    return g4tracks, pixnoise


def pyobjs_to_h5(g4tracks, pixnoise, filename, v):
    """
    Given the python object lists, write to HDF5 file at filename.
    """

    with h5py.File(filename, 'w', libver='latest', driver='core') as f:
        for ind in range(len(g4tracks)):
            pyobj_to_h5 = {}
            indstr = '{:05d}'.format(ind)
            g = f.create_group(indstr)

            # write g4track
            if g4tracks[ind]:
                filejob.vprint(
                    v, 3, 'Writing ' + str(ind) + ' g4track at ' +
                    time.ctime())
                trackio.write_object_to_hdf5(
                    g4tracks[ind], g, 'g4track', pyobj_to_h5=pyobj_to_h5)

            # write each pixnoise
            for key, val in pixnoise.iteritems():
                filejob.vprint(
                    v, 4, 'Writing ' + str(ind) + ' ' + key + ' at ' +
                    time.ctime())
                if isinstance(val[ind], trackdata.Track):
                    trackio.write_object_to_hdf5(
                        val[ind], g, key, pyobj_to_h5=pyobj_to_h5)
                else:
                    # track error code
                    t = g.create_group(key)
                    t.attrs['errorcode'] = val[ind]


if __name__ == '__main__':
    run_main()
