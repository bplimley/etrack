# compile_AR.py

from __future__ import print_function

import h5py
import os
import glob
import time
import ipdb as pdb

from etrack.reconstruction import evaluation
from etrack.io import trackio
from etrack.workspace.filejob import get_filename_function


def run_main():
    serverflag = True
    doneflag = True     # only take files that have an associated done file

    if serverflag:
        loadpath = '/global/home/users/bcplimley/multi_angle/HTbatch01_ARnew/'
        savepath = loadpath
    else:
        loadpath = '/media/plimley/TEAM 7B/HTbatch01_AR/'
        savepath = loadpath

    loadglob = 'MultiAngle_HT_*_*_AR.h5'
    savename = 'compile_AR_' + str(int(time.time()))
    doneglob = 'done2_MultiAngle_HT_*_*_AR.h5'

    if doneflag:
        flist = glob.glob(os.path.join(loadpath, doneglob))
        done2load = get_filename_function(doneglob, loadglob)
    else:
        flist = glob.glob(os.path.join(loadpath, loadglob))
    print('flist contains {} files'.format(str(len(flist))))

    pnlist = ['pix10_5noise0',
              'pix2_5noise0',
              'pix5noise0',
              'pix20noise0',
              'pix40noise0']
    alglist = ['python HT v1.5',
               'python HT v1.5a',
               'python HT v1.5b',
               'python HT v1.5c',
               'python HT v1.5d',
               'python HT v1.5e',
               'python HT v1.5f',
               'matlab HT v1.5']

    AR = {}
    for fname in flist:
        if doneflag:
            donename = os.path.split(fname)[-1]
            loadname = done2load(donename)
            loadfile = os.path.join(loadpath, loadname)
        else:
            loadfile = fname

        try:
            with h5py.File(loadfile, 'r') as h5f:
                for pn in pnlist:
                    if pn not in AR:
                        AR[pn] = {}
                    for alg in alglist:
                        try:
                            this_AR = evaluation.AlgorithmResults.from_hdf5(
                                h5f[pn][alg])
                        except KeyError:
                            # happens once somewhere
                            print('KeyError on {} - {} - {}'.format(
                                loadname, pn, alg))
                            continue
                        try:
                            AR[pn][alg] += this_AR
                        except KeyError:
                            AR[pn][alg] = this_AR
        except IOError:
            print('IOError on {}; skipping'.format(loadfile))

    print('saving to {}'.format(os.path.join(savepath, savename)))
    with h5py.File(os.path.join(savepath, savename), 'w') as fsave:
        for pn in pnlist:
            fsave.create_group(pn)
            for alg in alglist:
                AR[pn][alg].add_default_uncertainties()
                trackio.write_object_to_hdf5(AR[pn][alg], fsave[pn], alg)


if __name__ == '__main__':
    run_main()

    if False:
        pdb.set_trace()
        pass
