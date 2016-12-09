# tally_results.py
#
# tally results from compiled_results.h5 (result of compile_classify.py)
#
# Classification cases:
#  0: multiplicity / bad segmentation / corrupted file
#  1: escape. endpoint not found (or rejected)
#  2: escape. endpoint accepted. wrong end. both reject
#  3: escape. endpoint accepted. wrong end. moments accepts
#  4: escape. endpoint accepted. wrong end. ridge accepts
#  5: escape. endpoint accepted. wrong end. both accept
#  6: escape. endpoint accepted. right end. both reject
#  7: escape. endpoint accepted. right end. moments accepts
#  8: escape. endpoint accepted. right end. ridge accepts
#  9: escape. endpoint accepted. right end. both accept
# 10: contained. endpoint not found (or rejected)
# 11: contained. endpoint accepted. wrong end. both reject
# 12: contained. endpoint accepted. wrong end. moments accepts
# 13: contained. endpoint accepted. wrong end. ridge accepts
# 14: contained. endpoint accepted. wrong end. both accept
# 15: contained. endpoint accepted. right end. both reject. early scatter
# 16: contained. endpoint accepted. right end. both reject. no early scatter
# 17: contained. endpoint accepted. right end. moments accepts. early scatter
# 18: contained. endpoint accepted. right end. moments accepts. no early scattr
# 19: contained. endpoint accepted. right end. ridge accepts. early scatter
# 20: contained. endpoint accepted. right end. ridge accepts. no early scatter
# 21: contained. endpoint accepted. right end. both accept. early scatter
# 22: contained. endpoint accepted. right end. both accept. no early scatter

from __future__ import print_function
import numpy as np
import h5py
import os
import ipdb as pdb

from compile_classify import data_variable_list

TEST_KEY = 'energy_tot_kev'

# thresholds
ESCAPE_KEV = 2.0
MAX_END_MIN_KEV = 45.0
MIN_END_MAX_KEV = 25.0
PHI_MAX_DEG = 90
EDGE_PIXELS_MAX = 4
EDGE_SEGMENTS_MAX = 1


def get_filename():
    filepath = '/media/plimley/TEAM 7B/clresults_10.5_batch01'
    filename = 'compiled_results.h5'
    fullname = os.path.join(filepath, filename)
    return fullname


def main():
    filename = get_filename()
    datadict = get_data_dict(filename)
    datalen = get_datalen(datadict)
    datadict['case'] = np.ones(shape=(datalen,), dtype=int) * -1

    energy_lg = (datadict['energy_tot_kev'] > 100)
    print(' ')
    print('Data length: {}'.format(datalen))
    print('  Above 100keV: {}'.format(np.sum(energy_lg)))
    print(' ')

    n_tot = 0
    nE_tot = 0
    for n in xrange(23):
        cond_list = condition_lookup(n)
        this_lg = construct_logical(datadict, cond_list)
        datadict['case'][this_lg] = n

        this_n = np.sum(this_lg)
        this_nE = np.sum(this_lg & energy_lg)
        n_tot += this_n
        nE_tot += this_nE

        print('Case #{:2d}:   n = {:7d}      n(>100keV) = {:7d}'.format(
            n, this_n, this_nE))

    print('        n_tot = {:7d}  n(>100keV)_tot = {:7d}'.format(
        n_tot, nE_tot))


class Condition(object):
    """Simple container to contain the condition key, operator, and value."""

    def __init__(self, key, val, op=None):
        self.key = key
        self.value = val
        if op is None:
            self.op = '=='
        else:
            self.op = op


def construct_logical(datadict, cond_list):
    """
    Construct a logical vector based on a condition list.
    """

    lg = np.ones_like(datadict[TEST_KEY]).astype(bool)
    for cond in cond_list:
        lg = lg & (datadict[cond.key] == cond.value)

    return lg


def condition_lookup(casenum):
    """
    Conditions which describe one case number from the classification chart.
    """

    if casenum == 0:
        #  0: multiplicity / bad segmentation / corrupted file
        cond_list = [Condition('no_trk_error', 0)]
    else:
        cond_list = [Condition('no_trk_error', 1)]

    if casenum >= 1 and casenum <= 9:
        # escape
        cond_list.append(Condition('is_contained', 0))
    elif casenum > 0:
        # contained
        cond_list.append(Condition('is_contained', 1))

    if casenum == 1 or casenum == 10:
        # endpoint not found (or, not accepted)
        cond_list.append(Condition('endpoint_accept', 0))
    elif casenum > 0:
        # endpoint accepted
        cond_list.append(Condition('endpoint_accept', 1))

    if (casenum >= 2 and casenum <= 5) or (casenum >= 11 and casenum <= 14):
        # wrong end
        cond_list.append(Condition('wrong_end_flag', 1))
    elif casenum > 0:
        # right end
        cond_list.append(Condition('wrong_end_flag', 0))

    if casenum in (2, 6, 11, 15, 16):
        # both reject
        cond_list.append(Condition('ridge_accept', 0))
        cond_list.append(Condition('moments_accept', 0))
    elif casenum in (3, 7, 12, 17, 18):
        # moments accepts
        cond_list.append(Condition('ridge_accept', 0))
        cond_list.append(Condition('moments_accept', 1))
    elif casenum in (4, 8, 13, 19, 20):
        # ridge accepts
        cond_list.append(Condition('ridge_accept', 1))
        cond_list.append(Condition('moments_accept', 0))
    elif casenum in (5, 9, 14, 21, 22):
        # both accept
        cond_list.append(Condition('ridge_accept', 1))
        cond_list.append(Condition('moments_accept', 1))

    if casenum in (15, 17, 19, 21):
        # early scatter
        cond_list.append(Condition('early_scatter_flag', 1))
    elif casenum in (16, 18, 20, 22):
        # no early scatter
        cond_list.append(Condition('early_scatter_flag', 0))

    return cond_list


def get_data_dict(filename):
    """
    Load the hdf5 file and pull all the data from it into a dict.
    """

    varlist = data_variable_list()
    datadict = {}

    with h5py.File(filename, 'r') as f:
        datalen = get_datalen(f)
        for key in varlist:
            if key != 'filename':
                datadict[key] = np.empty(shape=(datalen,))
            else:
                datadict[key] = np.empty(shape=(datalen,), dtype='|S28')
            f[key].read_direct(datadict[key])

    # make implicit flags/parameters explicit
    datadict['no_trk_error'] = (datadict['trk_errorcode'] == 0).astype(int)
    datadict['is_contained'] = np.abs(datadict['energy_tot_kev'] -
                                      datadict['energy_dep_kev']) < ESCAPE_KEV
    datadict['endpoint_accept'] = (
        (datadict['max_end_energy_kev'] > MAX_END_MIN_KEV) &
        (datadict['min_end_energy_kev'] < MIN_END_MAX_KEV) &
        (datadict['n_ends'] > 0))
    datadict['moments_accept'] = (
        (np.abs(datadict['phi_deg']) < PHI_MAX_DEG) &
        (datadict['edge_pixels'] <= EDGE_PIXELS_MAX) &
        (datadict['edge_segments'] <= EDGE_SEGMENTS_MAX))
    datadict['ridge_accept'] = np.logical_not(
        np.isnan(datadict['alpha_ridge_deg']))
    return datadict


def get_datalen(datadict_like):
    """Get the length of the data vectors."""
    return datadict_like[TEST_KEY].shape[0]


if __name__ == '__main__':
    main()
