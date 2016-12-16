# tally_results.py
#
# tally results from compiled_results.h5 (result of compile_classify.py)
#
# Classification cases:
# [old#] [new#] [description]
#  0  0 multiplicity / bad segmentation / corrupted file
#  1  1 escape. endpoint not found
#     2 escape. endpoint rejected by min end (>25 keV)
#     3 escape. endpoint rejected by max end (<45 keV)
#     4 escape. endpoint rejected by min and max ends
#  2  5 escape. endpoint accepted. wrong end. both reject
#  3  6 escape. endpoint accepted. wrong end. moments accepts
#  4  7 escape. endpoint accepted. wrong end. ridge accepts
#  5  8 escape. endpoint accepted. wrong end. both accept
#  6  9 escape. endpoint accepted. right end. both reject
#  7 10 escape. endpoint accepted. right end. moments accepts
#  8 11 escape. endpoint accepted. right end. ridge accepts
#  9 12 escape. endpoint accepted. right end. both accept
# 10 13 contained. endpoint not found
#    14 contained. endpoint rejected by min end (>25 keV)
#    15 contained. endpoint rejected by max end (<45 keV)
#    16 contained. endpoint rejected by min and max ends
# 11 17 contained. endpoint accepted. wrong end. both reject
# 12 18 contained. endpoint accepted. wrong end. moments accepts
# 13 19 contained. endpoint accepted. wrong end. ridge accepts
# 14 20 contained. endpoint accepted. wrong end. both accept
# 15 21 contained. endpoint accepted. right end. both reject. early sc
# 16 22 contained. endpoint accepted. right end. both reject. no early sc
# 17 23 contained. endpoint accepted. right end. moments accepts. early sc
# 18 24 contained. endpoint accepted. right end. moments accepts. no early sc
# 19 25 contained. endpoint accepted. right end. ridge accepts. early sc
# 20 26 contained. endpoint accepted. right end. ridge accepts. no early sc
# 21 27 contained. endpoint accepted. right end. both accept. early sc
# 22 28 contained. endpoint accepted. right end. both accept. no early sc

from __future__ import print_function
import numpy as np
import h5py
import os
import ipdb as pdb

from compile_classify import data_variable_list
from make_bins import hardcoded_bins as get_bins

TEST_KEY = 'energy_tot_kev'
NUM_CASES = 29
SAVE_FILE = 'case_tally2.csv'

# thresholds
ESCAPE_KEV = 2.0
MAX_END_MIN_KEV = 45.0
MIN_END_MAX_KEV = 25.0
PHI_MAX_DEG = 90
EDGE_PIXELS_MAX = 4
EDGE_SEGMENTS_MAX = 1


def get_filename():
    filepath = '/home/plimley/gh/etrack/workspace'
    filename = 'compiled_results.h5'
    fullname = os.path.join(filepath, filename)
    return fullname


def main():
    filename = get_filename()
    datadict = get_data_dict(filename)

    n_tot, nE_tot = sort_cases(datadict)

    energy_bin_edges, beta_bin_edges = get_bins()

    matrix = construct_tally_matrix(datadict, energy_bin_edges, beta_bin_edges)

    write_csv(SAVE_FILE, matrix, energy_bin_edges, beta_bin_edges)


def sort_cases(datadict):
    datalen = get_datalen(datadict)
    datadict['case'] = np.ones(shape=(datalen,), dtype=int) * -1

    energy_lg = (datadict['energy_tot_kev'] > 100)
    print(' ')
    print('Data length: {}'.format(datalen))
    print('  Above 100keV: {}'.format(np.sum(energy_lg)))
    print(' ')

    n_tot = 0
    nE_tot = 0
    for n in xrange(NUM_CASES):
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

    return n_tot, nE_tot


def construct_tally_matrix(datadict, energy_bin_edges, beta_bin_edges):
    """
    Build a matrix of number of events, split by energy, beta, and case #.
    """

    if 'case' not in datadict:
        raise KeyError('Run sort_cases() before construct_tally_matrix()')

    matrix = np.zeros(shape=(
        len(energy_bin_edges) - 1,
        len(beta_bin_edges) - 1,
        NUM_CASES), dtype=int)

    for i in xrange(len(energy_bin_edges[:-1])):
        energy_lg = (
            (datadict['energy_tot_kev'] > energy_bin_edges[i]) &
            (datadict['energy_tot_kev'] <= energy_bin_edges[i + 1]))

        for j in xrange(len(beta_bin_edges[:-1])):
            beta_lg = (
                (datadict['beta_true_deg'] > beta_bin_edges[j]) &
                (datadict['beta_true_deg'] <= beta_bin_edges[j + 1]))

            for k in xrange(NUM_CASES):
                case_lg = (datadict['case'] == k)
                matrix[i, j, k] = np.sum(energy_lg & beta_lg & case_lg)

    return matrix


def show_event(datadict, n):
    """
    Display the relevant flags for a single event.
    """
    for flag in flag_list():
        value = datadict[flag][n]
        if type(value) is np.bool_:
            val = int(value)
        else:
            val = value
        print('{:20s}: {}'.format(flag, val))


def flag_list():
    """List of all the boolean flags used for cases"""
    flags = (
        'no_trk_error',
        'is_contained',
        'endpoint_accept',
        'wrong_end_flag',
        'ridge_accept',
        'moments_accept',
        'early_scatter_flag',
    )
    return flags


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

    if casenum >= 1 and casenum <= 12:
        # escape
        cond_list.append(Condition('is_contained', 0))
    elif casenum > 0:
        # contained
        cond_list.append(Condition('is_contained', 1))

    if casenum == 1 or casenum == 13:
        # endpoint not found
        cond_list.append(Condition('endpoint_found', 0))
    elif casenum == 2 or casenum == 14:
        # endpoint found, min end reject, max end accept
        cond_list.append(Condition('endpoint_found', 1))
        cond_list.append(Condition('min_end_accept', 0))
        cond_list.append(Condition('max_end_accept', 1))
    elif casenum == 3 or casenum == 15:
        # endpoint found, max end reject, min end accept
        cond_list.append(Condition('endpoint_found', 1))
        cond_list.append(Condition('min_end_accept', 1))
        cond_list.append(Condition('max_end_accept', 0))
    elif casenum == 4 or casenum == 16:
        # endpoint found, both max and min reject
        cond_list.append(Condition('endpoint_found', 1))
        cond_list.append(Condition('min_end_accept', 0))
        cond_list.append(Condition('max_end_accept', 0))
    elif casenum > 0:
        # endpoint found, both max and min accept
        cond_list.append(Condition('endpoint_found', 1))
        cond_list.append(Condition('min_end_accept', 1))
        cond_list.append(Condition('max_end_accept', 1))

    if (casenum >= 5 and casenum <= 8) or (casenum >= 17 and casenum <= 20):
        # wrong end
        cond_list.append(Condition('wrong_end_flag', 1))
    elif (casenum >= 9 and casenum <= 12) or casenum >= 21:
        # right end
        cond_list.append(Condition('wrong_end_flag', 0))

    if casenum in (5, 9, 17, 21, 22):
        # both reject
        cond_list.append(Condition('ridge_accept', 0))
        cond_list.append(Condition('moments_accept', 0))
    elif casenum in (6, 10, 19, 23, 24):
        # moments accepts
        cond_list.append(Condition('ridge_accept', 0))
        cond_list.append(Condition('moments_accept', 1))
    elif casenum in (7, 11, 19, 25, 26):
        # ridge accepts
        cond_list.append(Condition('ridge_accept', 1))
        cond_list.append(Condition('moments_accept', 0))
    elif casenum in (8, 12, 20, 27, 28):
        # both accept
        cond_list.append(Condition('ridge_accept', 1))
        cond_list.append(Condition('moments_accept', 1))

    if casenum in (21, 23, 25, 27):
        # early scatter
        cond_list.append(Condition('early_scatter_flag', 1))
    elif casenum in (22, 24, 26, 28):
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
    datadict['endpoint_found'] = (datadict['n_ends'] > 0)
    datadict['max_end_accept'] = (
        datadict['max_end_energy_kev'] > MAX_END_MIN_KEV)
    datadict['min_end_accept'] = (
        datadict['min_end_energy_kev'] < MIN_END_MAX_KEV)
    datadict['endpoint_accept'] = (
        datadict['endpoint_found'] &
        datadict['max_end_accept'] &
        datadict['min_end_accept'])
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


def write_csv(filename, matrix, energy_bins, beta_bins):
    """
    Write a CSV file with all the numbers in it.
    """

    with open(filename, 'w') as f:
        f.write(
            'beta min,beta max,' +
            ','.join(
                [str(c) for c in xrange(NUM_CASES)]) +
            '\n'
        )
        for i in xrange(len(energy_bins[:-1])):
            f.write(
                '\n,,{} keV < E_tot < {} keV:\n'.format(
                    energy_bins[i], energy_bins[i + 1]))

            for j in xrange(len(beta_bins[:-1])):
                f.write(
                    '{}deg,{}deg,'.format(
                        beta_bins[j], beta_bins[j + 1]) +
                    ','.join(
                        [str(el) for el in matrix[i, j, :]]) +
                    '\n'
                )


if __name__ == '__main__':
    main()
