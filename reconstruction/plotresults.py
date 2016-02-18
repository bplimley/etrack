# plotresults.py
#
# for handling AlgorithmResults objects and their metrics.

from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np

import evaluation
import trackio
import dataformats
from filejob import isstrlike


def plot_series(alg_results, vary='Etot', angle='alpha', metric=None,
                bin_edges=None, plot_kwargs=None):
    """
    Plot subsets of an evaluation.AlgorithmResults object, along one axis.

    Required input:
      alg_results: the evaluation.AlgorithmResults object to handle.

    Optional inputs:
      vary: Which variable to subdivide the results along.
        Allowed values: 'Etot', 'Edep', 'depth', 'beta_true', 'beta_meas'
        (default 'Etot')
      angle: 'alpha' or 'beta'. To avoid potential confusion re: metric.
        (default 'alpha')
      metric: 'FWHM' or 'f' for alpha, 'RMS' for beta.
        (default 'FWHM' for alpha, 'RMS' for beta)
      bin_edges: list of bin edges for the "vary" parameter.
        (default: depends on vary)

    """

    def input_handling(alg_results, vary, angle, metric, bin_edges,
                       plot_kwargs):
        # check for errors
        if not isinstance(alg_results, evaluation.AlgorithmResults):
            raise InputError(
                'plot_series: alg_results must be an AlgorithmResults object')
        if len(alg_results) < 10:
            raise InputError(
                'plot_series: insufficient data for plot_series()')

        if not isstrlike(vary):
            raise InputError(
                'plot_series: vary should be a string')
        if not isstrlike(angle):
            raise InputError(
                'plot_series: angle should be a string')
        if metric is not None and not isstrlike(metric):
            raise InputError(
                'plot_series: metric should be a string')
        if (bin_edges is not None and not isinstance(bin_edges, list) and
                not isinstance(bin_edges, tuple)):
            raise InputError(
                'plot_series: bin_edges should be a list or tuple')

        # options
        if vary.lower() == 'etot':
            vary = 'Etot'
            min_condition = 'energy_min'
            max_condition = 'energy_max'
        elif vary.lower() == 'edep':
            vary = 'Edep'
            raise InputError('plot_series: vary=Edep not supported')
            # because AlgorithmResults.select() doesn't support it yet
        elif vary.lower() == 'depth':
            vary = 'depth'
            min_condition = 'depth_min'
            max_condition = 'depth_max'
        elif vary.lower() == 'beta_true' or vary.lower() == 'b_true':
            vary = 'beta_true'
            min_condition = 'beta_true_min'
            max_condition = 'beta_true_max'
        elif vary.lower() == 'beta_meas' or vary.lower() == 'b_meas':
            vary = 'beta_meas'
            min_condition = 'beta_meas_min'
            max_condition = 'beta_meas_max'
        else:
            raise InputError(
                'plot_series: "vary" arg not recognized')

        if angle.lower() == 'alpha' or angle.lower() == 'a':
            angle = 'alpha'
        elif angle.lower() == 'beta' or angle.lower() == 'b':
            angle = 'beta'
        else:
            raise InputError(
                'plot_series: "angle" arg not recognized')

        # defaults
        if metric is None:
            if angle == 'alpha':
                metric = 'FWHM'
            else:
                metric = 'RMS'
        if metric.lower() == 'fwhm':
            metric = 'FWHM'
        elif metric.lower() == 'f':
            metric = 'f'
        elif metric.lower() == 'rms':
            metric = 'RMS'

        if bin_edges is None:
            if vary == 'Etot' or vary == 'Edep':
                bin_edges = range(0, 500, 50)
            elif vary == 'depth':
                bin_edges = range(0, 650, 50)
            elif vary.startswith('beta'):
                bin_edges = [0, 15, 30, 60]

        # calculate bin centers
        if len(bin_edges) < 2:
            raise InputError(
                'plot_series: bin_edges must be length at least 2')
        bin_centers = [(bin_edges[left] + bin_edges[left + 1]) / 2.0
                       for left in range(len(bin_edges)-1)]

        # TODO: add error checking on this
        if plot_kwargs is None:
            plot_kwargs = {}

        return (alg_results, vary, angle, metric, bin_edges, bin_centers,
                min_condition, max_condition, plot_kwargs)

    # main

    (alg_results, vary, angle, metric, bin_edges, bin_centers, min_condition,
     max_condition, plot_kwargs) = input_handling(
        alg_results, vary, angle, metric, bin_edges, plot_kwargs)

    vals = [[] for _ in range(len(bin_centers))]
    unc = [[] for _ in range(len(bin_centers))]

    for i, bin_center in enumerate(bin_centers):
        conditions = {min_condition: bin_edges[i],
                      max_condition: bin_edges[i + 1]}
        subset = alg_results.select(**conditions)
        if angle == 'alpha':
            subset.add_uncertainty(evaluation.DefaultAlphaUncertainty)
            this_metric = subset.alpha_unc.metrics[metric]
        else:
            subset.add_uncertainty(evaluation.DefaultBetaUncertainty)
            this_metric = subset.beta_unc.metrics[metric]
        vals[i] = this_metric.value
        unc[i] = this_metric.uncertainty[0]

    # print(vals)
    # print(unc)

    line = plt.errorbar(bin_centers, vals, yerr=unc, **plot_kwargs)
    plt.ylabel(angle + ' ' + metric)
    plt.xlabel(vary)
    plt.show()

    return line


class InputError(Exception):
    pass


def run_main():
    # 2016-02-17
    import h5py

    loadfile = '/media/plimley/TEAM 7B/HTbatch01_AR/compile_AR_1455774217'

    pnlist = ['pix10_5noise0', 'pix2_5noise0']
    alglist = ['python HT v1.5', 'python HT v1.5a', 'python HT v1.5b',
               'matlab HT v1.5']

    with h5py.File(loadfile, 'r') as h5f:
        AR = {}
        for pn in pnlist:
            AR[pn] = {}
            for alg in alglist:
                AR[pn][alg] = evaluation.AlgorithmResults.from_hdf5(
                    h5f[pn][alg])

    colors = ['k', 'b', 'r', 'g', 'c', 'm', '0.7', 'y']
    n = 0
    plt.figure()
    for pn in pnlist:
        for alg in alglist:
            kwargs = {'color': colors[n], 'label': pn + ' ' + alg}
            this_AR = AR[pn][alg].select(beta_true_max=30)

            plot_series(
                this_AR, bin_edges=range(0, 500, 100), plot_kwargs=kwargs)
            n += 1
    plt.legend()
    plt.show()


if __name__ == '__main__':
    run_main()
