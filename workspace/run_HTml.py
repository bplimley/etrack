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

import trackio
import trackdata
import evaluation
import hybridtrack


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
            if v > 2:
                print('    Starting', pix_key, 'at', time.ctime())
            this_h5pix = this_h5track[pix_key]
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
            HTout, HTinfo = hybridtrack.reconstruct(
                this_pixobj.image, pixel_size_um=this_pixobj.pixel_size_um)

            this_pixobj.add_algorithm(
                alg_name,
                alpha_deg=HTinfo.alpha_deg, beta_deg=HTinfo.beta_deg,
                info=HTinfo)
            if v > 2:
                print('    Appending to file', pix_key, 'at', time.ctime())
            # append to same file
            pdb.set_trace()
            if dry_run:
                continue
            trackio.write_object_to_hdf5(
                this_pixobj.algorithms[alg_name],
                this_h5pix['algorithms'], alg_name,
                pyobj_to_h5=pyobj_to_h5)



def run_main():
    location = 'extHD'
    if location == 'LRC':
        LOAD_DIR = '/global/home/users/bcplimley/multi_angle/HTbatch01_h5m/'
        # SAVE_DIR = '/global/home/users/bcplimley/multi_angle/HTbatch01_pyml/'
    elif location == 'desktop':
        # LBL desktop
        LOAD_DIR = ('/home/plimley/Documents/MATLAB/data/Electron Track/' +
                    'algorithms/results/2013sep binned')
        # SAVE_DIR = LOAD_DIR
    elif location == 'extHD':
        # 'TEAM 7B' WD USB 1TB hard drive
        LOAD_DIR = '/media/plimley/TEAM 7B/HTbatch01_pyml'
        # SAVE_DIR = '/media/plimley/TEAM 7B/HTbatch01_pyHT'
    else:
        raise RuntimeError('need a valid location')

    LOAD_FILE = 'MultiAngle_HT_*_*_py.h5'

    VERBOSITY = 4
    V = VERBOSITY       # for conciseness (=P)
    DRY_RUN = True

    for loadfile in glob.glob(os.path.join(LOAD_DIR, LOAD_FILE)):
        if V > 0: print('# Starting', loadfile, 'at', time.ctime(), '#')
        # savefile = os.path.join(SAVE_DIR, os.path.basename(loadfile))


        # if check_file(savefile): continue
        try:
            with h5py.File(loadfile, 'r+') as h5f:
                run_HT_file(h5f, V, DRY_RUN)
        except IOError:
            print('  !!! unable to load file', loadfile)
            continue

if __name__ == '__main__':
    run_main()
