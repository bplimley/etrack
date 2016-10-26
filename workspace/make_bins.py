
from __future__ import print_function

import os
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle


def bin_centers(bin_edges):
    """
    Compute a vector of bin centers from a vector of bin edges.
    """

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    return bin_centers


def load_energy_beta(fname):
    """
    Parse the text file from MATLAB to get arrays of Etot, Edep, beta.
    """

    with open(fname, 'r') as f:
        all_lines = f.readlines()

    for i, line in enumerate(all_lines):
        if line.startswith('Etot'):
            Etot_ind = i
            all_lines[i] = line[4:]
        elif line.startswith('Edep'):
            Edep_ind = i
            all_lines[i] = line[4:]
        elif line.startswith('beta'):
            beta_ind = i
            all_lines[i] = line[4:]

    Etot = np.array(all_lines[Etot_ind:Edep_ind], dtype=np.float)
    Edep = np.array(all_lines[Edep_ind:beta_ind], dtype=np.float)
    beta = np.array(all_lines[beta_ind:], dtype=np.float)

    return Etot, Edep, beta


def plot_beta_dists(Etot, beta):
    """
    Plot histograms of beta to check that they are independent of Etot.
    """

    E_bin_edges = np.array([0, 100, 250, 425, 440, 460, 500])
    E_bin_centers = bin_centers(E_bin_edges)

    plt.figure()
    ax = plt.axes()

    cycol = cycle('bgrcmk').next

    for i, center in enumerate(E_bin_centers):
        selection = (Etot > E_bin_edges[i]) & (Etot < E_bin_edges[i + 1])
        # make a histogram
        b_bin_edges = np.arange(-90, 91, 2)
        b_bin_centers = bin_centers(b_bin_edges)
        n, _ = np.histogram(beta[selection],
                            bins=b_bin_edges, density=True)
        # plot
        plt.plot(b_bin_centers, n,
                 drawstyle='steps-mid', c=cycol(),
                 label='{}-{}keV'.format(E_bin_edges[i], E_bin_edges[i + 1]))
        plt.legend()
        plt.xlim((-90, 90))
        plt.xlabel('beta [degrees]')
        plt.ylabel('fraction of events')


def find_energy_bins(Etot, n=8):
    """
    Find equal-area energy bins, except 0-100 keV. n is the number of bins.
    """

    s = np.sort(Etot)
    N = len(Etot)

    ind100 = np.nonzero(s > 100)[0][0]
    events_per_bin = float(N - ind100) / (n - 1)

    energy_bins = [0]
    ind_edges = np.arange(ind100, N+1, events_per_bin)
    for ind in ind_edges:
        try:
            energy_bins.append(s[ind])
        except IndexError:
            energy_bins.append(s[-1])

    return energy_bins


def find_beta_bins(n=10):
    """
    Find equal-area beta bins, assuming a cosine distribution.
    """

    # see logbook 10/25/2016
    # the PDF is a cosine. integrate to get a sine.
    # set sine equal to equally-spaced values.

    beta_bin_edges = 180 / np.pi * np.arcsin(np.arange(-1, 1, 0.2))

    return beta_bin_edges


def tally_bins(Etot, beta, energy_bins, beta_bins):
    """
    Tally the events per bin in the 2D binning of energy and beta.

    Bin vectors are bin edges from find_*_bins().
    """

    n = np.zeros((len(energy_bins) - 1, len(beta_bins) - 1))

    for i in xrange(len(energy_bins) - 1):
        for j in xrange(len(beta_bins) - 1):
            selection = (
                (Etot > energy_bins[i])
                & (Etot < energy_bins[i + 1])
                & (beta > beta_bins[j])
                & (beta < beta_bins[j + 1]))
            n[i, j] = np.sum(selection)

    return n


def plot_energy_bins(Etot, energy_bins):
    """
    Plot an energy histogram with the bins marked.

    energy_bins comes from find_energy_bins.
    """

    plt.figure()
    plt.axes()

    # energy spectrum
    dx = 2
    E_bin_edges = np.arange(0, 700, dx)
    E_bin_centers = bin_centers(E_bin_edges)
    n, _ = np.histogram(Etot, bins=E_bin_edges)
    plt.plot(E_bin_centers, n, 'k', drawstyle='steps-mid')

    # bin lines
    for x in energy_bins:
        y = n[int(np.round(x / dx))]
        plt.plot([x, x], [0, y], 'b')

    plt.xlabel('Energy [keV]')
    plt.ylabel('Tracks per {} keV'.format(dx))
    plt.title('Equal-area energy bins (except 0-100 keV)')
    plt.show()


def run_main():

    loadfile = 'energy_beta.txt'
    Etot, Edep, beta = load_energy_beta(loadfile)

    find_energy_bins(Etot)


if __name__ == '__main__':
    run_main()
