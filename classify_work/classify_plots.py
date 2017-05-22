
# -.- coding=utf8 -.-

"""Code for generating plots for 2017 moments/classification paper."""

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

from make_bins import hardcoded_bins as get_bins
import tally_results as tr


def main():
    energy_bin_edges, beta_bin_edges = get_bins()

    datadict = get_data_dict()

    plot_escape_fraction(datadict, energy_bin_edges, beta_bin_edges)


def get_data_dict():

    datadict = tr.get_data_dict(tr.get_filename())
    tr.sort_cases(datadict, write_in=True, verbose=False)
    return datadict


def plot_escape_fraction(datadict, e_bins, b_bins, abs_beta=True):
    """
    Escape fraction vs. energy
    Data series are different beta (or |beta|?)
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

        plt.errorbar(e_mean, f, yerr=f_unc, xerr=e_err, fmt='*',
                     label=u'{:.1f}° < β < {:.1f}°'.format(
                         b_bins[j], b_bins[j + 1]))

    plt.xlabel('Energy [keV]')
    plt.ylabel(u'Fraction escaping from 650 µm Si')
    plt.xlim((0, 500))
    plt.ylim((0, 0.6))
    plt.grid('on')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
