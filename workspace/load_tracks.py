# script for figuring out the 2.5um problem.

from __future__ import print_function
import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
import ipdb as pdb
import progressbar

import hybridtrack2f
import evaluation
import trackdata
import trackplot
import trackio
import plotresults


def run_main(g4=None, trks=None, info=None, ind=None):
    n_processes = 1
    T400flag = True
    tracks_only = True

    if T400flag:
        loadfile = '/mnt/data/Dropbox/MATLAB/MultiAngle_HT_100_11_py.h5'
    else:
        loadfile = os.path.join(
            '/media/plimley/TEAM 7B/HTbatch01_pyml/',
            'MultiAngle_HT_100_11_py.h5')

    if g4 is None or info is None or trks is None or ind is None:
        f = h5py.File(loadfile, 'r')

        target_list = []
        for key, val in f.iteritems():
            if 'g4track' in val:
                g4 = val['g4track']
                if 'energy_tot_kev' in g4.attrs and 'beta_deg' in g4.attrs:
                    if (g4.attrs['energy_tot_kev'] > 400 and
                            g4.attrs['beta_deg'] < 45):
                        target_list.append(key)

        print('{} tracks in target_list (energy and beta window)'.format(
            len(target_list)))
        print('Running target tracks...')

        if not tracks_only:
            info = {2: [], 5: []}
        g4 = []
        trks = {2: [], 5: []}
        ind = []

        if n_processes == 1:
            pbar = progressbar.ProgressBar(
                widgets=[progressbar.Percentage(), ' ', progressbar.Bar(), ' ',
                         progressbar.ETA()], maxval=len(target_list))
            pbar.start()
        for i, key in enumerate(target_list):
            if n_processes == 1:
                pbar.update(i)
            try:
                this_g4 = trackdata.G4Track.from_hdf5(f[key]['g4track'])
                this_trk2 = trackdata.Track.from_hdf5(f[key]['pix2_5noise0'])
                this_trk5 = trackdata.Track.from_hdf5(f[key]['pix5noise0'])
                if not tracks_only:
                    _, this_info2 = hybridtrack2f.reconstruct(this_trk2)
                    _, this_info5 = hybridtrack2f.reconstruct(this_trk5)
                ind.append(int(key))
                g4.append(this_g4)
                trks[2].append(this_trk2)
                trks[5].append(this_trk5)
                if not tracks_only:
                    info[2].append(this_info2)
                    info[5].append(this_info5)
            except KeyError:
                print('KeyError on {}'.format(key))
            except trackio.InterfaceError:
                print('InterfaceError on {}'.format(key))
        if n_processes == 1:
            pbar.finish()
    else:
        AR2all = evaluation.AlgorithmResults.from_track_list(trks[2])
        AR5all = evaluation.AlgorithmResults.from_track_list(trks[5])
        AR2all.add_default_uncertainties()
        AR5all.add_default_uncertainties()

        plt.figure()
        plotresults.plot_distribution(
            AR2all, density=True, bin_size=5, plot_kwargs={'color': 'b'})
        plotresults.plot_distribution(
            AR5all, density=True, bin_size=5, plot_kwargs={'color': 'k'})
        plt.show()

        interesting_tracks = []
        for i, g4t in enumerate(g4):
            try:
                da2 = np.abs(info[2][i].alpha_deg - g4t.alpha_deg)
                da5 = np.abs(info[5][i].alpha_deg - g4t.alpha_deg)
                # print('da2={}, da5={}'.format(str(da2), str(da5)))
                if da2 < 25 and da5 < 25 and da2 > da5:
                    # print('interesting! {}'.format(str(i)))
                    interesting_tracks.append(i)
            except AttributeError:
                continue
        print('{} interesting tracks (5um outperforms 2um)'.format(
            len(interesting_tracks)))

        for i in interesting_tracks:
            titletext = '#' + str(i) + ', {} um'
            trackplot.oneplot(info[2][i], g4=g4[i],
                              titletext=titletext.format('2.5'))
            trackplot.oneplot(info[5][i], g4=g4[i],
                              titletext=titletext.format('5'))
            raw_input()
            plt.close('all')

    if tracks_only:
        return g4, trks, ind
    else:
        return g4, trks, info, ind

if __name__ == '__main__':
    run_main()
    if False:
        pdb.set_trace()
