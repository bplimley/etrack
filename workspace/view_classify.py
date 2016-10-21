

import numpy as np
import os
import h5py
import socket
import matplotlib.pyplot as plt

import etrack.reconstruction.trackdata as td
import etrack.reconstruction.trackmoments as tm
import etrack.reconstruction.classify as cl
import etrack.visualization.trackplot as tp
import etrack.reconstruction.hybridtrack as ht
import etrack.reconstruction.evaluation as ev

DEG = '$^\circ$'


def run_main():
    """
    Go through the default file, show events based on user input.
    """

    loadfile = 'MultiAngle_algs_100_1.h5'
    loadfull = os.path.join(get_loadpath(), loadfile)
    prompt = '[#], [n]ext, [p]revious, [q]uit. [Enter] for next: '
    pn = 'pix10_5noise15'

    with h5py.File(loadfull, 'r') as f:
        usr_input = '0'
        fig = None

        while not usr_input.startswith('q'):
            skip = False

            # parse the input
            try:
                # assume it's a number
                indnum = int(usr_input)
            except ValueError:
                # it's a string command
                if usr_input == '' or usr_input.startswith('n'):
                    indnum += 1
                elif usr_input.startswith('p'):
                    indnum -= 1
                else:
                    print('### Unknown input')
                    skip = True
            indstr = '{:05d}'.format(indnum)

            # show the event
            if not skip:
                objs = get_event(f, indstr)
                try:
                    print('Event {}'.format(indstr))
                    fig = show_event(indstr, *objs)
                except KeyError:
                    print('### KeyError on {}/{}'.format(indstr, pn))

            # next input
            usr_input = raw_input(prompt)

            # close previous plot if exists
            try:
                plt.close(fig)
            except TypeError:
                # fig is None
                pass


def get_event(h5f, indstr):
    """
    Try building the Track, MomentsReconstruction, and Classifier of an event.
    """

    pn = 'pix10_5noise15'
    clpre = 'cl_'
    mompre = 'mom_'

    try:
        track = td.Track.from_hdf5(h5f[indstr][pn])
    except KeyError:
        track = None

    try:
        mom = tm.MomentsReconstruction.from_hdf5(h5f[mompre + indstr][pn])
    except KeyError:
        mom = None

    try:
        classifier = cl.Classifier.from_hdf5(h5f[clpre + indstr][pn])
    except KeyError:
        classifier = None

    return track, mom, classifier


def mom_reject(mom):
    """
    Return a rejection flag based on moments quantities, along with a reason.
    """

    phi_max = np.pi / 2
    edge_segments_max = 1
    edge_pixels_max = 4
    end_energy_max = 25.0

    reject = False  # until shown otherwise
    reason = ''

    if np.abs(mom.phi) > phi_max:
        reject = True
        reason += 'phi'
    if mom.edge_pixel_segments > edge_segments_max:
        reject = True
        if reason:
            reason += ','
        reason += 'segments'
    if mom.edge_pixel_count > edge_pixels_max:
        reject = True
        if reason:
            reason += ','
        reason += 'pixels'
    if mom.end_energy > end_energy_max:
        reject = True
        if reason:
            reason += ','
        reason += 'energy'

    return reject, reason


def plot_track(img):
    cmap = tp.get_colormap()
    plt.imshow(
        img, cmap=cmap,
        aspect='equal', interpolation='none', origin='lower')


def show_event(ind, track, mom, classifier):
    """
    Plot stuff for one event.

    geant4 is cyan
    ridge-following is light green
    moments is magenta
    """

    if track.g4track.energy_tot_kev < 100:
        print('Low energy event: {} keV'.format(track.g4track.energy_tot_kev))
        return None

    mom.reconstruct()
    try:
        _, HTinfo = ht.reconstruct(track)
    except ht.HybridTrackError:
        HTinfo = None

    # geant4 position
    g4x, g4y = tp.get_image_xy(track)
    g4x0 = np.array([g4x[0], g4y[0]])
    # geant4 direction (radians)
    g4_alpha_deg = track.g4track.alpha_deg
    g4_alpha_rad = g4_alpha_deg * np.pi / 180.0

    # moments
    mom_arc, mom_ep = tp.get_arc2(mom)
    mom_alpha_rad = mom.alpha
    reject_flag, reject_reason = mom_reject(mom)
    mom_da = ev.AlphaUncertainty.delta_alpha(
        g4_alpha_deg, mom_alpha_rad * 180.0 / np.pi)

    # ridge-following
    # HTinfo = track.algorithms['python HT v1.52'].info
    ridge_ep = (HTinfo.ridge[HTinfo.measurement_start_pt].coordinates_pix
                - np.array([1, 1]))
    ridge_alpha_rad = (track.algorithms['python HT v1.52'].alpha_deg
                       * np.pi / 180.0)
    ridge_da = ev.AlphaUncertainty.delta_alpha(
        g4_alpha_deg, ridge_alpha_rad * 180.0 / np.pi)

    titlebase = (
        r'{}' + '\n' +
        r'E={:.0f}keV, $\beta$={:.0f}{deg}, ' +
        r'scattering {:.1f}{deg}, E_end={:.1f}keV' + '\n' +
        r'Rejection: {}' + '\n')
    titlestr = titlebase.format(
        ind,
        track.g4track.energy_tot_kev,
        track.g4track.beta_deg,
        classifier.total_scatter_angle * 180 / np.pi,
        mom.end_energy,
        reject_reason,
        deg=DEG,)
    if np.abs(classifier.g4track.beta_deg) > 60:
        titlestr += '[Beta > 60{}]'.format(DEG)
    if classifier.early_scatter:
        titlestr += ' [Early scatter in 25um]'
    if classifier.overlap:
        titlestr += ' [Overlapping]'
    if classifier.wrong_end:
        titlestr += ' [Wrong end]'

    # PLOTS
    fig = plt.figure()

    # top left: ridge-following, zoomed
    plt.subplot(2, 2, 1)
    # g4
    plot_track(track.image)
    plt.plot(g4y, g4x, '.c')
    tp.plot_arrow(g4x0, g4_alpha_rad, color='c')
    # ridge
    ridge = HTinfo.ridge[HTinfo.measurement_start_pt:HTinfo.measurement_end_pt]
    tp.plot_ridgepoints(
        plt.gca(), ridge, fmtstring='.', offset=[1, 1], color=[0, 1, 0])
    tp.plot_arrow(ridge_ep, ridge_alpha_rad, color=[0, 1, 0])
    # general
    xoff = mom.end_segment_offsets[1]
    yoff = mom.end_segment_offsets[0]
    plt.xlim((xoff - 1, xoff + mom.end_segment_image.shape[1] + 1))
    plt.ylim((yoff - 1, yoff + mom.end_segment_image.shape[0] + 1))
    plt.xlabel('y [pixels]')
    plt.ylabel('x [pixels]')
    plt.colorbar()

    # bottom left: ridge-following, full
    plt.subplot(2, 2, 3)
    # g4
    plot_track(track.image)
    plt.plot(g4y, g4x, '.c')
    tp.plot_arrow(g4x0, g4_alpha_rad, color='c')
    # ridge
    ridge = HTinfo.ridge[HTinfo.measurement_start_pt:HTinfo.measurement_end_pt]
    tp.plot_ridgepoints(
        plt.gca(), ridge, fmtstring='.', offset=[1, 1], color=[0, 1, 0])
    tp.plot_arrow(ridge_ep, ridge_alpha_rad, color=[0, 1, 0])
    # general
    plt.xlim((0, track.image.shape[1] - 1))
    plt.ylim((0, track.image.shape[0] - 1))
    plt.xlabel('y [pixels]')
    plt.ylabel('x [pixels]')
    plt.colorbar()
    plt.title(r'Ridge-following, $\Delta_\alpha$ = {:2.1f}{}'.format(
        ridge_da, DEG))

    # top right: moments, zoomed
    plt.subplot(2, 2, 2)
    # g4
    plot_track(track.image)
    plt.plot(g4y, g4x, '.c')
    tp.plot_arrow(g4x0, g4_alpha_rad, color='c')
    # moments
    plt.plot(mom.box_y, mom.box_x, 'm')
    if reject_flag:
        plt.plot(mom_arc[1, :], mom_arc[0, :], 'm', lw=2.5, ls='dashed')
        tp.plot_arrow(mom_ep, mom_alpha_rad, color='m', ls='dashed')
    else:
        plt.plot(mom_arc[1, :], mom_arc[0, :], 'm', lw=2.5)
        tp.plot_arrow(mom_ep, mom_alpha_rad, color='m')
    # general
    xoff = mom.end_segment_offsets[1]
    yoff = mom.end_segment_offsets[0]
    plt.xlim((xoff - 1, xoff + mom.end_segment_image.shape[1] + 1))
    plt.ylim((yoff - 1, yoff + mom.end_segment_image.shape[0] + 1))
    plt.xlabel('y [pixels]')
    plt.ylabel('x [pixels]')
    plt.colorbar()

    # bottom right: moments, full
    plt.subplot(2, 2, 4)
    # g4
    plot_track(track.image)
    plt.plot(g4y, g4x, '.c')
    tp.plot_arrow(g4x0, g4_alpha_rad, color='c')
    # moments
    plt.plot(mom.box_y, mom.box_x, 'm')
    if reject_flag:
        plt.plot(mom_arc[1, :], mom_arc[0, :], 'm', lw=2.5, ls='dashed')
        tp.plot_arrow(mom_ep, mom_alpha_rad, color='m', ls='dashed')
    else:
        plt.plot(mom_arc[1, :], mom_arc[0, :], 'm', lw=2.5)
        tp.plot_arrow(mom_ep, mom_alpha_rad, color='m')
    # general
    plt.xlim((0, track.image.shape[1] - 1))
    plt.ylim((0, track.image.shape[0] - 1))
    plt.xlabel('y [pixels]')
    plt.ylabel('x [pixels]')
    plt.colorbar()
    plt.title(r'Moments, $\Delta_\alpha$ = {:2.1f}{}'.format(mom_da, DEG))

    fig.suptitle(titlestr)

    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.show()
    return fig


def get_loadpath():
    """
    Return a system-dependent path for the file to load.
    """

    if socket.gethostname() == 'plimley-Vostro-mint17':
        # LBL desktop
        loadpath = '/media/plimley/TEAM 7B/algs_10.5_batch01'
    elif socket.gethostname() == 'plimley-zenbook-mint':
        # laptop
        loadpath = '/home/plimley/Documents/research/data/algs_10.5_batch01'
    else:
        raise RuntimeError("I don't know what system I'm on")

    return loadpath


if __name__ == '__main__':
    run_main()
