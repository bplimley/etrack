# tally_results.py
#
# tally results from compiled_results.h5 (result of compile_classify.py)
#
# Classification cases:
# [old#] [new#] [description]
#  0  0 multiplicity / bad segmentation / corrupted file
#  1  1 escape. endpoint not found
#     2 escape. endpoint reject by min end (>25 keV). wrong end.
#     3 escape. endpoint reject by min end (>25 keV). right end. early sc
#     4 escape. endpoint reject by min end (>25 keV). right end. no early sc
#     5 escape. endpoint reject by max end (<45 keV). wrong end.
#     6 escape. endpoint reject by max end (<45 keV). right end. early sc
#     7 escape. endpoint reject by max end (<45 keV). right end. no early sc
#     8 escape. endpoint reject by min and max ends. wrong end.
#     9 escape. endpoint reject by min and max ends. right end. early sc
#    10 escape. endpoint reject by min and max ends. right end. no early sc
#  2 11 escape. endpoint accepted. wrong end. both reject
#  3 12 escape. endpoint accepted. wrong end. moments accepts
#  4 13 escape. endpoint accepted. wrong end. ridge accepts
#  5 14 escape. endpoint accepted. wrong end. both accept
#  6 15 escape. endpoint accepted. right end. both reject. early sc
#    16 escape. endpoint accepted. right end. both reject. no early sc
#  7 17 escape. endpoint accepted. right end. moments accepts. early sc
#    18 escape. endpoint accepted. right end. moments accepts. no early sc
#  8 19 escape. endpoint accepted. right end. ridge accepts. early sc
#    20 escape. endpoint accepted. right end. ridge accepts. no early sc
#  9 21 escape. endpoint accepted. right end. both accept. early sc
#    22 escape. endpoint accepted. right end. both accept. no early sc
# 10 23 contained. endpoint not found
#    24 contained. endpoint reject by min end (>25 keV). wrong end.
#    25 contained. endpoint reject by min end (>25 keV). right end. early sc
#    26 contained. endpoint reject by min end (>25 keV). right end. no early sc
#    27 contained. endpoint reject by max end (<45 keV). wrong end.
#    28 contained. endpoint reject by max end (<45 keV). right end. early sc
#    29 contained. endpoint reject by max end (<45 keV). right end. no early sc
#    30 contained. endpoint reject by min and max ends. wrong end.
#    31 contained. endpoint reject by min and max ends. right end. early sc
#    32 contained. endpoint reject by min and max ends. right end. no early sc
# 11 33 contained. endpoint accepted. wrong end. both reject
# 12 34 contained. endpoint accepted. wrong end. moments accepts
# 13 35 contained. endpoint accepted. wrong end. ridge accepts
# 14 36 contained. endpoint accepted. wrong end. both accept
# 15 37 contained. endpoint accepted. right end. both reject. early sc
# 16 38 contained. endpoint accepted. right end. both reject. no early sc
# 17 39 contained. endpoint accepted. right end. moments accepts. early sc
# 18 40 contained. endpoint accepted. right end. moments accepts. no early sc
# 19 41 contained. endpoint accepted. right end. ridge accepts. early sc
# 20 42 contained. endpoint accepted. right end. ridge accepts. no early sc
# 21 43 contained. endpoint accepted. right end. both accept. early sc
# 22 44 contained. endpoint accepted. right end. both accept. no early sc

from __future__ import print_function
import numpy as np
import h5py
import os
import ipdb as pdb

from compile_classify import data_variable_list
from make_bins import hardcoded_bins as get_bins

TEST_KEY = 'energy_tot_kev'
NUM_CASES = 45
SAVE_FILE = 'case_tally3.csv'

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

    if casenum >= 1 and casenum <= 22:
        # escape
        cond_list.append(Condition('is_contained', 0))
    elif casenum > 0:
        # contained
        cond_list.append(Condition('is_contained', 1))

    if casenum in (1, 23):
        # endpoint not found
        cond_list.append(Condition('endpoint_found', 0))
    elif casenum in (2, 3, 4, 24, 25, 26):
        # endpoint found, min end reject, max end accept
        cond_list.append(Condition('endpoint_found', 1))
        cond_list.append(Condition('min_end_accept', 0))
        cond_list.append(Condition('max_end_accept', 1))
    elif casenum in (5, 6, 7, 27, 28, 29):
        # endpoint found, max end reject, min end accept
        cond_list.append(Condition('endpoint_found', 1))
        cond_list.append(Condition('min_end_accept', 1))
        cond_list.append(Condition('max_end_accept', 0))
    elif casenum in (8, 9, 10, 30, 31, 32):
        # endpoint found, both max and min reject
        cond_list.append(Condition('endpoint_found', 1))
        cond_list.append(Condition('min_end_accept', 0))
        cond_list.append(Condition('max_end_accept', 0))
    elif casenum > 0:
        # endpoint found, both max and min accept
        cond_list.append(Condition('endpoint_found', 1))
        cond_list.append(Condition('min_end_accept', 1))
        cond_list.append(Condition('max_end_accept', 1))

    if casenum in (
            2, 5, 8,
            11, 12, 13, 14,
            24, 27, 30,
            33, 34, 35, 36):
        # wrong end
        cond_list.append(Condition('wrong_end_flag', 1))
    elif ((casenum in (3, 4, 6, 7, 9, 10)) or
            (casenum >= 15 and casenum <= 22) or
            (casenum in (25, 26, 28, 29, 31, 32)) or
            (casenum >= 37)):
        # right end
        cond_list.append(Condition('wrong_end_flag', 0))

    if casenum in (11, 15, 16, 33, 37, 38):
        # both algs reject
        cond_list.append(Condition('ridge_accept', 0))
        cond_list.append(Condition('moments_accept', 0))
    elif casenum in (12, 17, 18, 34, 39, 40):
        # moments accepts
        cond_list.append(Condition('ridge_accept', 0))
        cond_list.append(Condition('moments_accept', 1))
    elif casenum in (13, 19, 20, 35, 41, 42):
        # ridge accepts
        cond_list.append(Condition('ridge_accept', 1))
        cond_list.append(Condition('moments_accept', 0))
    elif casenum in (14, 21, 22, 36, 43, 44):
        # both algs accept
        cond_list.append(Condition('ridge_accept', 1))
        cond_list.append(Condition('moments_accept', 1))

    if casenum in (
            3, 6, 9,
            15, 17, 19, 21,
            25, 28, 31,
            37, 39, 41, 43):
        # early scatter
        cond_list.append(Condition('early_scatter_flag', 1))
    elif casenum in (
            4, 7, 10,
            16, 18, 20, 22,
            26, 29, 32,
            38, 40, 42, 44):
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
