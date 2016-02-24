
import h5py
import numpy as np
import matplotlib.pyplot as plt
import ipdb as pdb

import hybridtrack2f
import trackdata
import trackplot
import trackio

T400flag = False
if T400flag:
    loadfile = '/media/plimley/TEAM 7B/HTbatch01_pyml/MultiAngle_HT_100_11_py.h5'
else:
    loadfile = '/media/plimley/TEAM 7B/HTbatch01_pyml/MultiAngle_HT_100_11_py.h5'

f = h5py.File(loadfile, 'r')

target_list = []
for key, val in f.iteritems():
    if 'g4track' in val:
        g4 = val['g4track']
        if 'energy_tot_kev' in g4.attrs and 'beta_deg' in g4.attrs:
            if g4.attrs['energy_tot_kev'] > 400 and g4.attrs['beta_deg'] < 45:
                target_list.append(key)

g4 = []
info = {2: [], 5: []}
for key in target_list:
    try:
        this_g4 = f[key]['g4track']
        _, this_info2 = hybridtrack2f.reconstruct(trackdata.Track.from_hdf5(f[key]['pix2_5noise0']))
        _, this_info5 = hybridtrack2f.reconstruct(trackdata.Track.from_hdf5(f[key]['pix5noise0']))
        g4.append(this_g4)
        info[2].append(this_info2)
        info[5].append(this_info5)
    except KeyError:
        print('KeyError on {}'.format(key))
    except trackio.InterfaceError:
        print('InterfaceError on {}'.format(key))

interesting_tracks = []
for i, g4t in enumerate(g4):
    try:
        da2 = np.abs(info[2][i].alpha_deg - g4t.alpha_deg)
        da5 = np.abs(info[5][i].alpha_deg - g4t.alpha_deg)
        if da2 < 25 and da5 < 25 and da2 > da5:
            interesting_tracks.append(i)
    except AttributeError:
        continue
