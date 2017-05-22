
# -.- coding=utf8 -.-

"""Code for generating plots for 2017 moments/classification paper."""

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

from make_bins import hardcoded_bins as get_bins
import tally_results as tr

MKR = ('*', 's', 'o', 'x', '^')
MS = (8, 5, 5, 7, 6)

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=LARGE_SIZE)  # fontsize of the figure title
plt.rc('text', usetex=False)

arrowprops = {'width': 0, 'headwidth': 6, 'fc': 'k'}


def main():
    energy_bin_edges, beta_bin_edges = get_bins()

    datadict = get_data_dict()

    # plot_escape_fraction(datadict, energy_bin_edges, beta_bin_edges)
    # plot_early_scatter(datadict, energy_bin_edges, beta_bin_edges)
    plot_escape_roc(datadict, energy_bin_edges, beta_bin_edges)


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

    n_bins = len(b_bins) - 1
    if abs_beta:
        jrange = range(n_bins / 2, n_bins)
    else:
        jrange = range(n_bins)

    for j in jrange:
        if abs_beta:
            beta_lg = ((np.abs(datadict['beta_true_deg']) > b_bins[j]) &
                       (np.abs(datadict['beta_true_deg']) <= b_bins[j + 1]))
        else:
            beta_lg = ((datadict['beta_true_deg'] > b_bins[j]) &
                       (datadict['beta_true_deg'] <= b_bins[j + 1]))
        f = []
        f_unc = []
        e_mean = []
        e_err = np.zeros((2, len(e_bins) - 1))
        for i in range(len(e_bins) - 1):
            e_lg = ((datadict['energy_tot_kev'] > e_bins[i]) &
                    (datadict['energy_tot_kev'] <= e_bins[i + 1]))
            n_esc = np.sum(beta_lg & e_lg & (datadict['is_contained'] == 0))
            n_con = np.sum(beta_lg & e_lg & (datadict['is_contained'] == 1))
            n_tot = n_esc + n_con
            f.append(float(n_esc) / float(n_tot))
            f_unc.append(np.sqrt(n_tot * f[-1] * (1 - f[-1])) / float(n_tot))
            e_mean.append(np.mean(datadict['energy_tot_kev'][e_lg & beta_lg]))
            e_err[0, i] = e_mean[-1] - e_bins[i]
            e_err[1, i] = e_bins[i + 1] - e_mean[-1]

        m = MKR[j % len(MKR)]
        ms = MS[j % len(MS)]
        if not yerr:
            f_unc = None
        plt.errorbar(e_mean, f, yerr=f_unc, xerr=e_err, fmt=m, ms=ms,
                     label=u'{:.1f}° < β < {:.1f}°'.format(
                         b_bins[j], b_bins[j + 1]))

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

    n_bins = len(b_bins) - 1
    if abs_beta:
        jrange = range(n_bins / 2, n_bins)
    else:
        jrange = range(n_bins)

    for j in jrange:
        if abs_beta:
            beta_lg = ((np.abs(datadict['beta_true_deg']) > b_bins[j]) &
                       (np.abs(datadict['beta_true_deg']) <= b_bins[j + 1]))
        else:
            beta_lg = ((datadict['beta_true_deg'] > b_bins[j]) &
                       (datadict['beta_true_deg'] <= b_bins[j + 1]))
        f = []
        f_unc = []
        e_mean = []
        e_err = np.zeros((2, len(e_bins) - 1))
        for i in range(len(e_bins) - 1):
            e_lg = ((datadict['energy_tot_kev'] > e_bins[i]) &
                    (datadict['energy_tot_kev'] <= e_bins[i + 1]))
            n_sc = np.sum(
                beta_lg & e_lg & (datadict['early_scatter_flag'] == 1))
            n_no = np.sum(
                beta_lg & e_lg & (datadict['early_scatter_flag'] == 0))
            n_tot = n_sc + n_no
            f.append(float(n_sc) / float(n_tot))
            f_unc.append(np.sqrt(n_tot * f[-1] * (1 - f[-1])) / float(n_tot))
            e_mean.append(np.mean(datadict['energy_tot_kev'][e_lg & beta_lg]))
            e_err[0, i] = e_mean[-1] - e_bins[i]
            e_err[1, i] = e_bins[i + 1] - e_mean[-1]

        m = MKR[j % len(MKR)]
        ms = MS[j % len(MS)]
        if not yerr:
            f_unc = None
        plt.errorbar(e_mean, f, yerr=f_unc, xerr=e_err, fmt=m, ms=ms,
                     label=u'{:.1f}° < β < {:.1f}°'.format(
                         b_bins[j], b_bins[j + 1]))

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
    conf_list_esc = []
    print('Building escape ROC curve with {} points'.format(
        len(test_thresholds_esc)), end='')
    for thresh in test_thresholds_esc:
        caselist = tr.sort_cases(datadict, max_end_min_kev=thresh)
        this_conf = tr.ConfusionMatrix.from_cases(
            caselist, tr.ESCAPE_CASE_DICT, name='escape', thresh=thresh)
        conf_list_esc.append(this_conf)
        print('.', end='')
    roc_esc = tr.RocCurve.from_confmat_list(conf_list_esc)

    roc_esc.plot()


def get_roc_data():
    return tr.roc_curves()


def low_end_roc_curve(roc_data):
    """Plot the ROC curve for the low-energy end."""

    roc, conf_list, test_thresholds = roc_data[0], roc_data[2], roc_data[4]
    fig = make_roc_plot(roc, conf_list, test_thresholds, label='low')
    plt.savefig('roc_low.png')


def escape_roc_curve(roc_data):
    """Plot the ROC curve for the escape."""

    roc, conf_list, test_thresholds = roc_data[1], roc_data[3], roc_data[5]
    fig = make_roc_plot(roc, conf_list, test_thresholds, label='escape')
    plt.savefig('roc_esc.png')


def make_roc_plot(roc, conf_list, test_thresholds, label=None):
    """Plot the ROC curve for the low-energy end."""

    fig = plt.figure(figsize=(8, 8))
    ax = roc.plot(color='C0', lw=2, label=label)
    ax.grid('on')
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

    return fig


if __name__ == '__main__':
    main()
