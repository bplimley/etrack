# runalgorithm.py
#
# 2/24/2016. For running different algorithm on a small set of tracks,
# for fixing the 2.5um issue.

import numpy as np
import ipdb as pdb
import progressbar
import time


def runalgorithm(track_list, reconstruct_list):
    """
    Inputs:
      track_list: list of m tracks to process
      reconstruct_list: list of n reconstruct functions to run on each track

    Output:
      np.array (m x n) of da results.
    """

    progressflag = True

    maxm = len(track_list)
    maxn = len(reconstruct_list)
    result = np.zeros((maxm, maxn))
    errorcount = np.zeros((maxn,))

    if progressflag:
        pbar = progressbar.ProgressBar(
            widgets=[progressbar.Percentage(), ' ', progressbar.Bar(), ' ',
                     progressbar.ETA()], maxval=maxm * maxn)
        pbar.start()
    for n, reconstruct in enumerate(reconstruct_list):
        t1 = time.time()
        for m, track in enumerate(track_list):
            if progressflag:
                pbar.update(n * maxm + m)
            try:
                _, this_info = reconstruct(track)
                this_da = this_info.alpha_deg - track.g4track.alpha_deg
                result[m, n] = this_da
            except IOError:
                # placeholder except block
                result[m, n] = np.nan
                errorcount[n] += 1
        t2 = time.time()
        print(
            'Algorithm {} finished in {} s with {} errors for {} s/track'.format(
            reconstruct, t2 - t1, errorcount[n], (t2 - t1) / (maxm - errorcount[n])
            ))

    if progressflag:
        pbar.finish()

    return result
