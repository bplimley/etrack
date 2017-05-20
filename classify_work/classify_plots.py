"""Code for generating plots for 2017 moments/classification paper."""

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

from compile_classify import data_variable_list
from make_bins import hardcoded_bins as get_bins
import tally_results as tr


def main():
    energy_bin_edges, beta_bin_edges = get_bins()

    datadict = get_data_dict()


def get_data_dict():

    datadict = tr.get_data_dict(tr.get_filename())
    sort_cases(datadict, write_in=True, verbose=True)


def plot_escape_fraction(datadict, e_bins, b_bins):
    """
    Escape fraction vs. energy
    Data series are different beta (or |beta|?)
    """

    plt.figure(figsize=(8, 6))

    for j in range(len(b_bins) - 1):
        beta_lg = ((datadict['beta_true_deg'] > b_bins[j]) &
                   (datadict['beta_true_deg'] <= b_bins[j + 1]))
        f = []
        f_unc = []
        for i in range(len(e_bins) - 1):
            n_esc = np.sum(beta_lg & datadict['is_contained'] == 0)
            n_con = np.sum(beta_lg & datadict['is_contained'] == 1)
            n_tot = n_esc + n_con
            f.append(float(n_esc) / float(n_tot))
            f_unc.append(np.sqrt(n_tot * f[-1] * (1 - f[-1])))

        # plt.plot
