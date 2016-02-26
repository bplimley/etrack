# plotresults.py
#
# for handling AlgorithmResults objects and their metrics.

from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import ipdb as pdb

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


def plot_distribution(alg_results, angle='alpha', bin_size=None,
                      density=None, plot_kwargs=None):
    """
    Plot the distribution of an AlgorithmResults object.
    """

    # TODO: input handling
    if plot_kwargs is None:
        plot_kwargs = {}
    if density is None:
        density = False

    if angle == 'alpha':
        delta = evaluation.AlphaUncertainty.delta_alpha(
            alg_results.alpha_true_deg, alg_results.alpha_meas_deg)
        histrange = (-180, 180)
        if bin_size is None:
            n_bins = np.ceil(4 * len(alg_results)**(1./3.))
        else:
            n_bins = np.ceil(360.0 / bin_size)
        if n_bins % 2 == 1:
            # odd number is better, gives a peak at 0
            n_bins += 1

    hist, bin_edges = np.histogram(delta, range=histrange, bins=n_bins)
    hist = np.array([float(h) for h in hist])
    if density:
        # pdb.set_trace()
        histsum = np.sum(hist)
        hist /= histsum
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    histerr = np.sqrt(hist)
    if density:
        histerr /= histsum

    stepsflag = True
    errflag = False
    if stepsflag:
        drawstyle = 'steps-mid'
    else:
        drawstyle = 'default'
    if errflag:
        plt.errorbar(bin_centers, hist, yerr=histerr, drawstyle=drawstyle,
                     **plot_kwargs)
    else:
        plt.plot(bin_centers, hist, drawstyle=drawstyle, **plot_kwargs)

    plt.ylabel('N')
    plt.xlabel(angle)
    plt.xlim(histrange)
    plt.show()

    if density:
        return histsum
    else:
        return None


def fwhm_vs_energy(AR, bin_edges=None, only_contained=True):
    """
    get the FWHM vs. E of an AlgorithmResults instance
    """

    if bin_edges=None:
        bin_edges = range(50, 500, 50)
    bin_centers = float(bin_edges[1:] + bin_edges[:-1]) / 2

    if only_contained:
        AR = AR.select(is_contained=True)

    fwhm = np.zeros((len(bin_centers),))
    fwhm_unc = np.zeros((len(bin_centers),))
    for i in xrange(len(bin_centers)):
        energy_min = bin_edges[i]
        energy_max = bin_edges[i + 1]
        this_AR = AR.select(energy_min=energy_min, energy_max=energy_max)
        this_AR.add_default_uncertainties()
        fwhm[i] = this_AR.alpha_unc.metrics['fwhm'].value
        fwhm_unc[i] = this_AR.alpha_unc.metrics['fwhm'].uncertainty[0]

    return fwhm, fwhm_unc, bin_centers


def compare_AR(titletext=None, **kwargs):
    """
    Input key-value pairs, where key is a label (e.g. algname)
    and value is the AR object (including all energies)

    Plot the residuals.
    """

    only_contained = True
    bin_edges = range(50, 500, 25)
    colors = ['k', 'b', 'r', 'g', 'c', 'm', '0.7', 'y']

    fwhm = {}
    fwhm_unc = {}
    for algname, this_AR in kwargs.iteritems():
        fwhm[algname], fwhm_unc[algname], x = fwhm_vs_energy(
            this_AR, bin_edges=bin_edges, only_contained=only_contained)
    mean_values = [np.sum([f[i] for f in fwhm.values()])
                   / length(fwhm) for i in xrange(len(x))]
    residuals = {}
    for algname in fwhm.keys():
        residuals[algname] = fwhm[algname] - mean_values

    plt.figure()
    n = 0
    for algname in fwhm.keys():
        plt.errorbar(x, fwhm[algname], err=fwhm_unc[algname],
                     color=colors[n % len(colors)], label=algname)
    plt.xlabel('Energy [keV]')
    plt.ylabel('FWHM [degrees]')
    plt.legend()
    if titletext:
        plt.title(titletext)
    plt.show()


def compare_algorithms(ARdict):
    """
    Input:
      ARdict has algorithm results in ARdict[pn][alg]
    """

    pnlist, alglist = get_lists()
    algdict = {}

    for pn in pnlist:
        for alg in alglist:
            algdict[alg] = ARdict[pn][alg]
        compare_AR(titletext=pn, **algdict)


def temp0(AR):
    # plot vs pixelsize and energy
    pnlist, alglist = get_lists()

    # pnlist = pnlist[:2]
    # pnlist = [pnlist[2]]
    # alglist = alglist[:-2]
    alglist = [alglist[1], alglist[3], alglist[4], alglist[5]]

    colors = ['k', 'b', 'r', 'g', 'c', 'm', '0.7', 'y']
    # colors = ['c', 'm', '0.7', 'y']
    # colors = 'b' * 6 + 'r' * 6
    n = 0
    plt.figure()
    for pn in pnlist:
        for alg in alglist:
            this_color = colors[n % len(colors)]
            kwargs = {'color': this_color, 'label': pn + ' ' + alg}
            this_AR = AR[pn][alg].select(is_contained=True)

            plot_series(
                this_AR, bin_edges=range(50, 500, 25), plot_kwargs=kwargs)
            n += 1
    plt.legend()
    plt.ylim((0, 90))
    plt.show()


def temp1(alg_results, titletext=''):
    # plot distributions by energy
    bin_edges = np.array([100, 200, 300, 400, 500])
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2.0

    colors = ['k', 'b', 'r', 'g', 'm', 'c', 'y', '0.7']
    plt.figure()
    for i, bin_center in enumerate(bin_centers):
        Emin = bin_edges[i]
        Emax = bin_edges[i + 1]
        labeltext = str(Emin) + 'keV < E < ' + str(Emax) + 'keV'
        this_AR = alg_results.select(
            energy_min=Emin, energy_max=Emax)
        this_AR.add_default_uncertainties()
        this_FWHM = this_AR.alpha_unc.metrics['FWHM'].value
        this_f = this_AR.alpha_unc.metrics['f'].value
        labeltext += ', FWHM={:2.1f}, f={:2.1f}%'.format(this_FWHM, this_f)
        plot_distribution(
            this_AR, density=True,
            plot_kwargs={'color': colors[i], 'label': labeltext})
    plt.legend()
    plt.title(titletext)
    plt.show()


class InputError(Exception):
    pass


def get_lists(pnlist=None, alglist=None):
    pnlist = ['pix2_5noise0',
              'pix5noise0',
              'pix10_5noise0',
              'pix20noise0',
              'pix40noise0']
    alglist = ['matlab HT v1.5',
               'python HT v1.5',
               'python HT v1.5c',
               'python HT v1.5d',
               'python HT v1.5e',
               'python HT v1.5f',
               'python HT v1.5b',
               'python HT v1.5a']

    return pnlist, alglist


def run_main():
    # 2016-02-17
    import h5py

    loadfile = '/media/plimley/TEAM 7B/HTbatch01_AR/compile_AR_1456347532'

    pnlist, alglist = get_lists()

    with h5py.File(loadfile, 'r') as h5f:
        AR = {}
        for pn in pnlist:
            AR[pn] = {}
            for alg in alglist:
                # if pn.startswith('pix2_5noise0') and alg == 'python HT v1.5d':
                #     pdb.set_trace()
                AR[pn][alg] = evaluation.AlgorithmResults.from_hdf5(
                    h5f[pn][alg])

    temp0(AR)

    # ax = plt.axes()
    # this_AR = AR['pix10_5noise0']['matlab HT v1.5'].select(
    #     energy_min=250, energy_max=300, is_contained=True)
    # this_AR.add_default_uncertainties()
    # print('  FWHM: {:2.1f} +- {:1.1f}'.format(
    #     this_AR.alpha_unc.metrics['FWHM'].value,
    #     this_AR.alpha_unc.metrics['FWHM'].uncertainty[0]))
    # print('  f:    {:2.1f} +- {:1.1f}'.format(
    #     this_AR.alpha_unc.metrics['f'].value,
    #     this_AR.alpha_unc.metrics['f'].uncertainty[0]))
    # histsum = plot_distribution(this_AR, bin_size=3, density=True)
    #
    # # xx = np.linspace(-180, 180, num=3600)
    # # yfit = 2 / histsum * this_AR.alpha_unc.fit.eval(x=xx)
    # # plt.plot(xx, yfit, 'r', lw=2, label='python fit')
    # plt.title('250 keV < E < 300 keV, contained electrons, Matlab alg v1.5')
    # plt.show()
    #
    # this_AR.alpha_unc.fit.plot()
    # plt.show()
    #
    # # temp1(AR['pix10_5noise0']['matlab HT v1.5'], titletext='10um matlab')


if __name__ == '__main__':
    run_main()