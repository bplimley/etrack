# compile_AR.py

from __future__ import print_function

import h5py
import os
import glob
import time
import ipdb as pdb

import evaluation
import trackio


def run_main():
    serverflag = True
    if serverflag:
        loadpath = '/global/home/users/bcplimley/multi_angle/HTbatch01_ARnew/'
        savepath = loadpath
    else:
        loadpath = '/media/plimley/TEAM 7B/HTbatch01_AR/'
        savepath = loadpath
    loadglob = 'MultiAngle_HT_*_*_AR.h5'
    savename = 'compile_AR_' + str(int(time.time()))

    flist = glob.glob(os.path.join(loadpath, loadglob))
    print('flist contains {} files'.format(str(len(flist))))

    pnlist = ['pix10_5noise0',
              'pix2_5noise0',
              'pix5noise0',
              'pix20noise0',
              'pix40noise0']
    alglist = ['python HT v1.5',
               'python HT v1.5a',
               'python HT v1.5c',
               'matlab HT v1.5']

    AR = {}
    for fname in flist:
        with h5py.File(fname, 'r') as h5f:

            for pn in pnlist:
                if pn not in AR:
                    AR[pn] = {}
                for alg in alglist:
                    try:
                        this_AR = evaluation.AlgorithmResults.from_hdf5(
                            h5f[pn][alg])
                    except ZeroDivisionError:
                        continue
                    try:
                        AR[pn][alg] += this_AR
                    except KeyError:
                        AR[pn][alg] = this_AR

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
