
# -.- coding=utf8 -.-

"""Code for generating plots for 2017 moments/classification paper."""

from __future__ import print_function
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

from make_bins import hardcoded_bins as get_bins
import tally_results as tr
import etrack.reconstruction.evaluation as ev

MKR = ('*', 's', 'o', 'x', '^')
MS = (8, 5, 5, 7, 6)

SMALL_SIZE = 14
MEDIUM_SIZE = 18
LARGE_SIZE = 24

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=LARGE_SIZE)  # fontsize of the figure title
plt.rc('text', usetex=False)

arrowprops = {'width': 0, 'headwidth': 6, 'fc': 'k'}

savepath = '/home/plimley/gh/ETCI/papers/moments/'


def main():
    energy_bin_edges, beta_bin_edges = get_bins()

    datadict = get_data_dict()

    plot_escape_fraction(datadict, energy_bin_edges, beta_bin_edges)
    plt.savefig(os.path.join(savepath, 'f_esc.eps'))

    plot_early_scatter(datadict, energy_bin_edges, beta_bin_edges)
    plt.savefig(os.path.join(savepath, 'f_earlysc.eps'))

    plot_escape_roc(datadict, energy_bin_edges, beta_bin_edges)
    plt.savefig(os.path.join(savepath, 'roc_escape.eps'))

    plot_ridge_alpha(datadict, energy_bin_edges, beta_bin_edges)


def get_data_dict():

    datadict = tr.get_data_dict(tr.get_filename())
    tr.sort_cases(datadict, write_in=True, verbose=False)
    return datadict


def plot_escape_fraction(datadict, e_bins, b_bins, abs_beta=True, yerr=False):
    """
    Escape fraction vs. energy

    abs_beta: if True, combine pos and neg beta bins
    yerr: if False, assume y error bars are smaller than the markers, and skip
    """

    plt.figure(figsize=(8, 6))

    energy_lg, e_left, e_right = get_energy_lg(datadict, e_bins)
    beta_lg, b_left, b_right = get_beta_lg(datadict, b_bins, abs_beta=abs_beta)
    if abs_beta:
        labelbase = u'{:.1f}° < |β| < {:.1f}°'
    else:
        labelbase = u'{:.1f}° < β < {:.1f}°'

    for j, b_lg in enumerate(beta_lg):
        f = []
        f_unc = []
        e_mean = []
        e_err = np.zeros((2, len(e_bins) - 1))
        for i, e_lg in enumerate(energy_lg):
            n_esc = np.sum(b_lg & e_lg & (datadict['is_contained'] == 0))
            n_con = np.sum(b_lg & e_lg & (datadict['is_contained'] == 1))
            n_tot = n_esc + n_con
            f.append(float(n_esc) / float(n_tot))
            f_unc.append(np.sqrt(n_tot * f[-1] * (1 - f[-1])) / float(n_tot))
            e_mean.append(np.mean(datadict['energy_tot_kev'][e_lg & b_lg]))
            e_err[0, i] = e_mean[-1] - e_left[i]
            e_err[1, i] = e_right[i] - e_mean[-1]
            if i + 1 == len(energy_lg):
                print(u'f_escape for |β| {}-{}°, E {}-{} keV: {}±{}'.format(
                    b_left[j], b_right[j], e_left[i], e_right[i],
                    f[-1], f_unc[-1]))

        m = MKR[j % len(MKR)]
        ms = MS[j % len(MS)]
        if not yerr:
            f_unc = None
        plt.errorbar(e_mean, f, yerr=f_unc, xerr=e_err, fmt=m, ms=ms,
                     label=labelbase.format(b_left[j], b_right[j]))

    plt.xlabel('Energy [keV]')
    plt.ylabel(u'Fraction escaping from 650 µm Si')
    plt.xlim((0, 500))
    plt.ylim((0, 0.6))
    plt.grid('on')
    plt.legend()
    plt.show()


def plot_early_scatter(datadict, e_bins, b_bins, abs_beta=True, yerr=False):
    """
    Early scatter vs. energy

    abs_beta: if True, combine pos and neg beta bins
    yerr: if False, assume y error bars are smaller than the markers, and skip
    """

    plt.figure(figsize=(8, 6))

    energy_lg, e_left, e_right = get_energy_lg(datadict, e_bins)
    beta_lg, b_left, b_right = get_beta_lg(datadict, b_bins, abs_beta=abs_beta)
    if abs_beta:
        labelbase = u'{:.1f}° < |β| < {:.1f}°'
    else:
        labelbase = u'{:.1f}° < β < {:.1f}°'

    for j, b_lg in enumerate(beta_lg):
        f = []
        f_unc = []
        e_mean = []
        e_err = np.zeros((2, len(energy_lg)))
        for i, e_lg in enumerate(energy_lg):
            n_sc = np.sum(b_lg & e_lg & (datadict['early_scatter_flag'] == 1))
            n_no = np.sum(b_lg & e_lg & (datadict['early_scatter_flag'] == 0))
            n_tot = n_sc + n_no
            f.append(float(n_sc) / float(n_tot))
            f_unc.append(np.sqrt(n_tot * f[-1] * (1 - f[-1])) / float(n_tot))
            e_mean.append(np.mean(datadict['energy_tot_kev'][e_lg & b_lg]))
            e_err[0, i] = e_mean[-1] - e_left[i]
            e_err[1, i] = e_right[i] - e_mean[-1]

        m = MKR[j % len(MKR)]
        ms = MS[j % len(MS)]
        if not yerr:
            f_unc = None
        plt.errorbar(e_mean, f, yerr=f_unc, xerr=e_err, fmt=m, ms=ms,
                     label=labelbase.format(b_left[j], b_right[j]))

    plt.xlabel('Energy [keV]')
    plt.ylabel(u'Fraction scattering >30° in <25µm 2D')
    plt.xlim((0, 500))
    plt.ylim((0, 1))
    plt.grid('on')
    plt.legend()
    plt.show()


def plot_escape_roc(datadict, e_bins, b_bins, abs_beta=True):
    """Plot ROC for rejecting escape electrons."""

    # based on tally_results.roc_curves()
    test_thresholds_esc = np.arange(20, 101)
    conf_list_esc = np.empty(
        (len(e_bins) - 1, len(test_thresholds_esc)), dtype='O')

    e_lg, e_left, e_right = get_energy_lg(datadict, e_bins)

    print('Building escape ROC curve with {} points'.format(
        len(test_thresholds_esc)), end='')

    for j, thresh in enumerate(test_thresholds_esc):
        caselist = tr.sort_cases(datadict, max_end_min_kev=thresh)
        for i, lg in enumerate(e_lg):
            this_conf = tr.ConfusionMatrix.from_cases(
                caselist[lg], tr.ESCAPE_CASE_DICT, thresh=thresh)
            conf_list_esc[i, j] = this_conf
        print('.', end='')
        sys.stdout.flush()

    print(' ')
    plt.figure(figsize=(10, 8))
    ax = plt.axes()
    for i in range(len(e_lg)):
        this_roc = tr.RocCurve.from_confmat_list(conf_list_esc[i, :])
        this_roc.plot(ax=ax, lw=2, color='C' + str(i), mark=45 - 20,
                      label='{}-{} keV'.format(e_left[i], e_right[i]))
        if i + 2 == len(e_bins):
            make_roc_plot(this_roc, conf_list_esc[i, :], test_thresholds_esc,
                          label='{}-{} keV'.format(e_left[i], e_right[i]))

    plt.legend()
    plt.grid('on')
    plt.show()


# ~~~~~~~~~~~~~~~~~~~~~~~


def get_roc_data():
    """OUTDATED"""
    return tr.roc_curves()


def low_end_roc_curve(roc_data):
    """OUTDATED - Plot the ROC curve for the low-energy end."""

    roc, conf_list, test_thresholds = roc_data[0], roc_data[2], roc_data[4]
    fig = make_roc_plot(roc, conf_list, test_thresholds, label='low')
    plt.savefig('roc_low.png')


def escape_roc_curve(roc_data):
    """ OUTDATED - Plot the ROC curve for the escape."""

    roc, conf_list, test_thresholds = roc_data[1], roc_data[3], roc_data[5]
    fig = make_roc_plot(roc, conf_list, test_thresholds, label='escape')
    plt.savefig('roc_esc.png')


def make_roc_plot(roc, conf_list, test_thresholds, label=None):
    """Plot the threshold labels for an ROC plot."""

    # fig = plt.figure(figsize=(8, 8))
    # ax = roc.plot(color='C0', lw=2, label=label)
    # ax.grid('on')
    ax = plt.gca()
    # plt.plot([0, 1], [0, 1], ':k', lw=2)

    if len(test_thresholds) > 50:
        interval = 10
    else:
        interval = 5
    label_indices = np.arange(0, len(test_thresholds), interval).astype(int)

    for ind in label_indices:
        xydata = (roc.FPR[ind], roc.TPR[ind])
        xytext = (xydata[0] + 0.12, xydata[1] - 0.12)
        ax.annotate('{} keV'.format(test_thresholds[ind]),
                    xy=xydata, xytext=xytext, arrowprops=arrowprops,
                    size=14)

    # ind = np.flatnonzero(
    #     test_thresholds == tr.DEFAULT_MIN_END_MAX_KEV)[0]
    # confmat = conf_list[ind]
    # plt.plot(confmat.FPR, confmat.TPR, '*g', ms=12, lw=3)

    plt.xlabel('False Positive Rate (track discarded erroneously)',
               fontsize=15)
    plt.ylabel('True Positive Rate (track discarded correctly)', fontsize=15)


def plot_ridge_alpha(datadict, e_bins, b_bins):
    """..."""

    labelbase = u'{:.1f}° < |β| < {:.1f}°'

    beta_lg, b_left, b_right = get_beta_lg(
        datadict, b_bins, abs_beta=True)
    energy_lg, e_left, e_right = get_energy_lg(datadict, e_bins)

    ridge = np.empty((len(beta_lg), len(energy_lg)), dtype='O')
    moments = np.empty((len(beta_lg), len(energy_lg)), dtype='O')

    for j, b_lg in enumerate(beta_lg):
        for i, e_lg in enumerate(energy_lg):
            this_ridge_ar = ev.AlgorithmResults.from_datadict(
                datadict, b_lg & e_lg, 'ridge')
            this_moments_ar = ev.AlgorithmResults.from_datadict(
                datadict, b_lg & e_lg, 'moments')
            this_ridge_ar.add_uncertainty(ev.AlphaGaussPlusConstant)
            this_moments_ar.add_uncertainty(ev.AlphaGaussPlusConstant)
            ridge[j, i] = this_ridge_ar
            moments[j, i] = this_moments_ar

    algs = ('ridge', 'moments')
    for k, ar in enumerate((ridge, moments)):
        for m, metric in enumerate(('FWHM', 'f')):
            plt.figure(figsize=(8, 6))
            ax = plt.axes()
            for j, b_lg in enumerate(beta_lg):
                mk = MKR[j % len(MKR)]
                ms = MS[j % len(MS)]
                metrics = []
                e_mean = []
                e_err = np.zeros((2, len(energy_lg)))
                for i, e_lg in enumerate(energy_lg):
                    e_mean.append(
                        np.mean(datadict['energy_tot_kev'][e_lg & b_lg]))
                    e_err[0, i] = e_mean[-1] - e_left[i]
                    e_err[1, i] = e_right[i] - e_mean[-1]
                    metrics.append(ar[j, i].alpha_unc.metrics[metric])
                y = [met.value for met in metrics]
                yerr = [met.uncertainty[0] for met in metrics]
                ax.errorbar(e_mean, y, yerr=yerr, xerr=e_err, fmt=mk, ms=ms,
                            label=labelbase.format(b_left[j], b_right[j]))
            plt.xlabel('Energy [keV]')
            plt.ylabel('{} [{}]'.format(metrics[0].name, metrics[0].units))
            plt.ylim((metrics[0].axis_min, metrics[0].axis_max))
            plt.xlim((0, 500))
            plt.title('{} {}'.format(algs[k], metrics[0].name))
            plt.legend()
            plt.grid('on')
            plt.show()
            savename = '{}_{}.png'.format(('FWHM', 'f')[m], algs[k])
            plt.savefig(os.path.join(savepath, savename))


def get_beta_lg(datadict, b_bins, abs_beta=True):
    """Get a list of logical vectors, representing each beta bin."""

    n_bins = len(b_bins) - 1
    if abs_beta:
        jrange = range(n_bins / 2, n_bins)
    else:
        jrange = range(n_bins)

    beta_lg = []
    beta_left = []
    beta_right = []

    for j in jrange:
        if abs_beta:
            this_lg = ((np.abs(datadict['beta_true_deg']) > b_bins[j]) &
                       (np.abs(datadict['beta_true_deg']) <= b_bins[j + 1]))
        else:
            this_lg = ((datadict['beta_true_deg'] > b_bins[j]) &
                       (datadict['beta_true_deg'] <= b_bins[j + 1]))
        beta_lg.append(this_lg)
        beta_left.append(b_bins[j])
        beta_right.append(b_bins[j + 1])

    return beta_lg, beta_left, beta_right


def get_energy_lg(datadict, e_bins):
    """Get a list of logical vectors, representing each energy bin."""

    energy_lg = []
    energy_left = []
    energy_right = []
    for i in range(len(e_bins) - 1):
        this_lg = ((datadict['energy_tot_kev'] > e_bins[i]) &
                   (datadict['energy_tot_kev'] <= e_bins[i + 1]))
        energy_lg.append(this_lg)
        energy_left.append(e_bins[i])
        energy_right.append(e_bins[i + 1])

    return energy_lg, energy_left, energy_right


if __name__ == '__main__':
    main()
