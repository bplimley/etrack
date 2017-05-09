# generate plots for Moments and Classification paper, May 2017

from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

import tally_results as tr

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
