# run_HTml.py
#
# comes after h5matlab_to_py.py
#
# loads all the py object format h5 files, runs HybridTrack, saves.

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
import hybridtrack2


def check_file(checkfile):
    """
    See if checkfile exists, return true if it does, false otherwise.
    """
    return os.path.isfile(checkfile)


def run_HT_file(h5f, v, dry_run):
    """
    Handle one file: h5f['00000']['pix10_5noise0'] and so
    """

    for track_key in h5f.keys():
        # h5save.create_group(track_key)
        if v > 1:
            print('  Starting', track_key, 'at', time.ctime())
        this_h5track = h5f[track_key]
        h5_to_pydict = {}
        pydict_to_pyobj = {}
        pyobj_to_h5 = {}

        pix_list = ['pix2_5noise0', 'pix10_5noise0']
        if (pix_list[0] not in this_h5track.keys() or
                pix_list[1] not in this_h5track.keys()):
            continue
        for pix_key in pix_list:
            # if not pix_key.startswith('pix'):
            #     continue

            alg_name = 'python HT v1.5'
            alg_name2 = 'python HT v1.5a'
            if v > 2:
                print('    Starting', pix_key, 'at', time.ctime())
            this_h5pix = this_h5track[pix_key]
            if 'algorithms' not in this_h5pix.keys():
                continue
            if alg_name in this_h5pix['algorithms'].keys():
                continue
            if v > 3:
                print('      Reading pydict...', time.ctime())
            this_dict = trackio.read_object_from_hdf5(
                this_h5pix, h5_to_pydict=h5_to_pydict)
            if v > 3:
                print('      Building Track object...', time.ctime())
            this_pixobj = trackdata.Track.from_pydict(
                this_dict, pydict_to_pyobj=pydict_to_pyobj)
            if v > 3:
                print('      Running HybridTrack...', time.ctime())
            try:
                HTout, HTinfo = hybridtrack.reconstruct(
                    this_pixobj.image, pixel_size_um=this_pixobj.pixel_size_um)
                HTout2, HTinfo2 = hybridtrack2.reconstruct(
                    this_pixobj.image, pixel_size_um=this_pixobj.pixel_size_um)
            except hybridtrack.HybridTrackError, hybridtrack2.HybridTrackError:
                if v > 1:
                    print('      HybridTrackError on', track_key,
                          'at', time.ctime())
                continue

            this_pixobj.add_algorithm(
                alg_name,
                alpha_deg=HTinfo.alpha_deg, beta_deg=HTinfo.beta_deg,
                info=HTinfo)
            this_pixobj.add_algorithm(
                alg_name2,
                alpha_deg=HTinfo2.alpha_deg, beta_deg=HTinfo2.beta_deg,
                info=HTinfo2)
            if v > 2:
                print('    Appending to file', pix_key, 'at', time.ctime())
            # append to same file
            if dry_run:
                continue
            trackio.write_object_to_hdf5(
                this_pixobj.algorithms[alg_name],
                this_h5pix['algorithms'], alg_name,
                pyobj_to_h5=pyobj_to_h5)
            trackio.write_object_to_hdf5(
                this_pixobj.algorithms[alg_name2],
                this_h5pix['algorithms'], alg_name2,
                pyobj_to_h5=pyobj_to_h5)


def do_work(loadfile):
    v, dry_run = verbosity_dry_run()

    if v > 0:
        print('# Starting', loadfile, 'at', time.ctime(), '#')
    # savefile = os.path.join(SAVE_DIR, os.path.basename(loadfile))

    # if check_file(savefile): continue
    try:
        with h5py.File(loadfile, 'r+') as h5f:
            run_HT_file(h5f, v, dry_run)
    except IOError:
        print('  !!! unable to load file', loadfile)

    return None


def verbosity_dry_run():
    v = 1
    dry_run = False
    return v, dry_run


def run_main():
    location = 'extHD'

    if location == 'LRC':
        LOAD_DIR = '/global/home/users/bcplimley/multi_angle/HTbatch01_h5m/'
        # SAVE_DIR = '/global/home/users/bcplimley/multi_angle/HTbatch01_pyml/'
        n_threads = 11
    elif location == 'desktop':
        # LBL desktop
        LOAD_DIR = ('/home/plimley/Documents/MATLAB/data/Electron Track/' +
                    'algorithms/results/2013sep binned')
        # SAVE_DIR = LOAD_DIR
        n_threads = 7
    elif location == 'extHD':
        # 'TEAM 7B' WD USB 1TB hard drive
        LOAD_DIR = '/media/plimley/TEAM 7B/HTbatch01_pyml'
        # SAVE_DIR = '/media/plimley/TEAM 7B/HTbatch01_pyHT'
        n_threads = 7
    else:
        raise RuntimeError('need a valid location')

    LOAD_FILE = 'MultiAngle_HT_*_*_py.h5'

    V, DRY_RUN = verbosity_dry_run()

    flist = glob.glob(os.path.join(LOAD_DIR, LOAD_FILE))

    p = multiprocessing.Pool(n_threads)
    p.map(do_work, flist)


if __name__ == '__main__':
    run_main()
