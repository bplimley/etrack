# runalgorithm.py
#
# 2/24/2016. For running different algorithm on a small set of tracks,
# for fixing the 2.5um issue.

import numpy as np
import ipdb as pdb
import progressbar
import time

from etrack.reconstruction.evaluation import AlgorithmResults


def runalgorithm(track_list, reconstruct_dict):
    """
    Inputs:
      track_list: list of m tracks to process
      reconstruct_list: list of n reconstruct MODULES to run on each track

    Output:
      np.array (m x n) of da results.
    """

    progressflag = True

    maxm = len(track_list)
    maxn = len(reconstruct_dict)
    result = np.zeros((maxm, maxn))
    errorcount = np.zeros((maxn,))
    skipcount = np.zeros((maxn,))

    if progressflag:
        pbar = progressbar.ProgressBar(
            widgets=[progressbar.Percentage(), ' ', progressbar.Bar(), ' ',
                     progressbar.ETA()], maxval=maxm * maxn)
        pbar.start()
    for n, alg_name in enumerate(reconstruct_dict.keys()):
        t1 = time.time()
        for m, track in enumerate(track_list):
            if progressflag:
                pbar.update(n * maxm + m)
            if alg_name in track.list_algorithms():
                skipcount[n] += 1
            else:
                try:
                    _, this_info = reconstruct_dict[alg_name].reconstruct(track)
                    track.add_algorithm(
                        alg_name, this_info.alpha_deg, this_info.beta_deg)
                    this_da = this_info.alpha_deg - track.g4track.alpha_deg
                    result[m, n] = this_da
                except reconstruct_dict[alg_name].InfiniteLoop:
                    result[m, n] = np.nan
                    errorcount[n] += 1
                except reconstruct_dict[alg_name].NoEndsFound:
                    result[m, n] = np.nan
                    errorcount[n] += 1
        t2 = time.time()
        print(
            ('Algorithm {} finished in {} s ' +
             'with {} errors for {} s/track').format(
                alg_name, t2 - t1, errorcount[n],
                (t2 - t1) / (maxm - errorcount[n] - skipcount[n])
            ))

    if progressflag:
        pbar.finish()

    return result


def construct_ARdict(track_list, alglist):
    """
    Make a dict of algorithms (NOT pixsize/noise!), from the track list.
    """

    ARdict = {}
    for alg in alglist:
        ARdict[alg] = AlgorithmResults.from_track_list(
            track_list, alg_name=alg)

    return ARdict


if False:
    pdb.set_trace()
    pass
