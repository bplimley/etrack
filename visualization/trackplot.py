import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.patches as patches
import matplotlib
import scipy.interpolate

import ipdb as pdb

# get RidgePoint and Cut classes
# import hybridtrack


def oneplot(HTinfo, g4=None, titletext=None):
    """
    """

    if g4 is None:
        g4flag = False
    else:
        g4flag = True

    img = HTinfo.prepared_image_kev
    ridge = HTinfo.ridge

    ax, im = plot_track_image(img)

    # highlight measurement points
    start_pt = HTinfo.measurement_start_pt
    end_pt = HTinfo.measurement_end_pt

    this_ridge = ridge[:start_pt]
    [plot_best_cut(ax, r) for r in this_ridge]
    plot_ridgepoints(ax, this_ridge)

    this_ridge = ridge[start_pt:end_pt]
    [plot_best_cut(ax, r, fmtstring='g') for r in this_ridge]
    plot_ridgepoints(ax, this_ridge, fmtstring='g.')

    this_ridge = ridge[end_pt:]
    if this_ridge:
        [plot_best_cut(ax, r) for r in this_ridge]
        plot_ridgepoints(ax, this_ridge)

    # plot arrows
    plot_alpha_arrow(ax, HTinfo, fmtstring='g')
    if g4flag:
        plot_alpha_arrow(ax, HTinfo, alpha=g4.alpha_deg, fmtstring='m')
    if titletext is not None:
        plt.title(titletext)
    plt.show()

    return plt.gcf()


def plot_track_image(img):
    """
    """

    cmap = get_colormap()
    ax = plt.axes()
    im = plt.imshow(
        img, cmap=cmap, aspect='equal', interpolation='none', origin='lower')

    return ax, im


def plot_ridgepoints(ax, ridge_points, fmtstring='c.', **kwargs):
    """
    """

    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()

    coordinates = np.array([r.coordinates_pix for r in ridge_points])

    pts = plt.plot(coordinates[:, 1],
                   coordinates[:, 0],
                   fmtstring, axes=ax, **kwargs)

    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)

    return pts


def plot_best_cut(ax, ridge_point, fmtstring='c', **kwargs):
    """
    """

    best_cut = ridge_point.cuts[ridge_point.best_ind]
    c = plot_cut(ax, best_cut, fmtstring, **kwargs)

    return c


def plot_cut(ax, cut, fmtstring='c', **kwargs):
    """
    """

    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()

    coordinates = np.array(cut.coordinates_pix)
    c = plt.plot(coordinates[:, 1],
                 coordinates[:, 0],
                 fmtstring,
                 axes=ax,
                 **kwargs)

    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)

    return c


def plot_alpha_arrow(ax, HTinfo,
                     alpha=None, fmtstring='c', arrow_length=5, **kwargs):
    """
    Plot an arrow indicating an alpha direction.
    arrow_length is in pixels.
    """

    # linewidth
    if not kwargs:
        kwargs = {'lw': 3}
    arrow_length = float(arrow_length)
    if alpha is None:
        alpha = HTinfo.alpha_deg / 180 * np.pi
    else:
        alpha = float(alpha) / 180 * np.pi
    init_pt = HTinfo.ridge[HTinfo.measurement_start_pt].coordinates_pix

    side_angle = 30.0 / 180 * np.pi
    side_length = arrow_length / 5
    x0 = [0,
          arrow_length,
          arrow_length - side_length * np.cos(side_angle),
          arrow_length,
          arrow_length - side_length * np.cos(side_angle)]
    y0 = [0,
          0,
          side_length * np.sin(side_angle),
          0,
          -side_length * np.sin(side_angle)]
    x0, y0 = np.array(x0), np.array(y0)

    x = init_pt[0] + x0 * np.cos(alpha) - y0 * np.sin(alpha)
    y = init_pt[1] + x0 * np.sin(alpha) + y0 * np.cos(alpha)

    a = plt.plot(y, x, fmtstring, axes=ax, **kwargs)

    return a


def plot_moments_segment(mom):
    """
    Plot a track image and draw a box where trackmoments wants to work.

    box is [min_x, max_x, min_y, max_y]
    """

    f = plt.figure()
    ax, im = plot_track_image(mom.original_image_kev)

    plt.plot(mom.box_y, mom.box_x, 'c', axes=ax)
    # plt.plot(mom.ypix, mom.xpix, '*c', ms=8)

    return f


def plot_clist_circles(clist):
    """
    Plot a CoordinatesList object, using different sized circles depending on
    the total weighting at each coordinate.
    """

    # copies
    xc = clist.x
    yc = clist.y
    Ec = clist.E

    xnew = np.zeros_like(xc)
    ynew = np.zeros_like(xc)
    Enew = np.zeros_like(xc)

    n = 0
    for i in xrange(len(xc)):
        try:
            find_existing = np.nonzero((xnew == xc[i]) & (ynew == yc[i]))[0][0]
        except IndexError:
            xnew[n] = xc[i]
            ynew[n] = yc[i]
            Enew[n] = Ec[i]
            n += 1
        else:
            Enew[find_existing] += Ec[i]

    # normalize energy to 1
    Enew = np.sqrt(Enew) / np.sqrt(np.max(Enew))

    # make plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in xrange(len(xnew)):
        ax.add_patch(patches.Circle(
            (xnew[i], ynew[i]), radius=Enew[i]/2, fill=True, alpha=0.4))
        # plt.plot(xnew[i], ynew[i], 'ok', markersize=Enew[i])

    plt.xlim((np.min(xnew) - 1, np.max(xnew) + 1))
    plt.ylim((np.min(ynew) - 1, np.max(ynew) + 1))

    # pdb.set_trace()
    # pass


def plot_moments_arc(mom, debug=False, end_segment=False, box=False,
                     entry_pt=True, title=None):
    """
    Plot the end segment image, with the arc calculated by moments overlaid.
    """

    # needs to be migrated to use e.g. mom.rotated_to_full(xy)

    phi2 = np.abs(mom.phi) / 2
    # get center point of arc
    center_in_rotated_frame = [0, -mom.R * np.sin(phi2) / phi2]
    phi0 = np.pi / 2 - phi2
    phi1 = np.pi / 2 + phi2

    if debug:
        # figure 1
        plot_clist_circles(mom.clist2)
        plot_arc(center_in_rotated_frame, mom.R, phi0, phi1, flipxy=False)

    # rotation_angle was how the *coordinate frame* was rotated.
    # so to reverse it, rotate the *points* by the same amount.
    new_rotation_angle = mom.rotation_angle
    center_in_central_frame = [
        -center_in_rotated_frame[1] * np.sin(new_rotation_angle),
        center_in_rotated_frame[1] * np.cos(new_rotation_angle)]
    phi0 += new_rotation_angle
    phi1 += new_rotation_angle

    if debug:
        # figure 2
        plot_clist_circles(mom.clist1)
        plot_arc(center_in_central_frame, mom.R, phi0, phi1, flipxy=False)

    center_in_segment_frame = (
        np.array(center_in_central_frame) +
        np.array([mom.xoffset, mom.yoffset]))
    if debug:
        # figure 3
        plot_clist_circles(mom.clist0)
        plot_arc(center_in_segment_frame, mom.R, phi0, phi1, flipxy=False)

    # offset is good to know anyway (see `if end_segment` / `if box` below)
    imgoffset = np.array([np.min(mom.box_x), np.min(mom.box_y)])
    imgoffset[imgoffset < 0] = 0
    if not end_segment:
        # gotta go back to the original image too
        center_in_image_frame = (
            center_in_segment_frame + mom.end_segment_offsets)

    f = plt.figure()
    if end_segment:
        ax, im = plot_track_image(mom.end_segment_image)
        plot_arc(center_in_segment_frame, mom.R, phi0, phi1)
        if box:
            plt.plot(mom.box_y - imgoffset[1], mom.box_x - imgoffset[0], 'c')
    else:
        ax, im = plot_track_image(mom.original_image_kev)
        plot_arc(center_in_image_frame, mom.R, phi0, phi1)
        if box:
            plt.plot(mom.box_y, mom.box_x, 'c')
    if entry_pt:
        # testing
        plt.plot(mom.arc_center[1], mom.arc_center[0], '*b', lw=3, ms=10)
        plt.plot(mom.x0[1], mom.x0[0], '^c', lw=3, ms=10)
    if title is not None:
        plt.title(title)
    plt.show()

    if debug:
        import ipdb as pdb
        pdb.set_trace()
    return f


def plot_g4points(track):
    """
    Plot track image with 2D Geant4 positions overlaid.

    For testing offset corrections for plot_moments_track...
    """

    g4x, g4y = get_image_xy(track)

    # actual plotting
    plt.figure()
    ax, im = plot_track_image(track.image)

    plt.plot(g4y, g4x, '.c')

    plt.xlim((0, track.image.shape[1] - 1))
    plt.ylim((0, track.image.shape[0] - 1))


def get_image_xy(track):
    """
    Find the xy in image coordinates, of the Geant4 track.

    Uses find_g4_offsets.

    Returns x, y.
    """

    g4x_um = track.g4track.x
    img = track.image

    minx = np.min(g4x_um[0, :])
    miny = np.min(g4x_um[1, :])
    # maxx = np.max(g4x_um[0, :])
    # maxy = np.max(g4x_um[1, :])

    try:
        pixsize = np.float(track.pixel_size_um)
    except TypeError:
        pixsize = 10.5

    # align to a corner of pixel grid - "more minimum" than (minx, miny)
    minx_aligned = minx - (minx % pixsize)
    miny_aligned = miny - (miny % pixsize)

    # buffer is 8.5 pixels in ViewGeantTrack4.m. probably this is wrong
    buf = 4.5 * pixsize

    xoff = minx_aligned - buf
    yoff = miny_aligned - buf

    g4x_pix = np.array([(g4x_um[0, :] - xoff) / pixsize,
                        (g4x_um[1, :] - yoff) / pixsize])

    # check for bad track - like a brems interaction off in the distance
    g4size = np.array([np.max(g4x_pix[0, :]) - np.min(g4x_pix[0, :]),
                       np.max(g4x_pix[1, :]) - np.min(g4x_pix[1, :])])
    if g4size[0] > img.shape[0] or g4size[1] > img.shape[1]:
        raise G4TrackTooBigError

    xoff2, yoff2 = find_g4_offsets(g4x_pix, img)

    g4x_pix[0, :] -= xoff2
    g4x_pix[1, :] -= yoff2

    return g4x_pix[0, :], g4x_pix[1, :]


def find_g4_offsets(xfull, img):
    """
    x are the xy coordinates of a g4 track, in units of pixels.
    img is the track image.

    Move x around (by units of 1 pixel) until it fits on img best.
    """

    decimation = 10     # don't use all the points. 1 in 10 should be enough
    x = xfull[:, ::decimation]

    maxoffset = 5      # pixels

    # brute force...
    xoffs = range(-maxoffset, maxoffset)
    yoffs = range(-maxoffset, maxoffset)
    energy_sum = np.zeros((len(xoffs), len(yoffs)))

    size = img.shape
    interp = scipy.interpolate.RectBivariateSpline(
        range(size[0]), range(size[1]), img, kx=1, ky=1)

    for ix in xrange(len(xoffs)):
        for iy in xrange(len(yoffs)):
            thisx = x[0, :] - xoffs[ix]
            thisy = x[1, :] - yoffs[iy]
            energy_sum[ix, iy] = compute_energy_sum(thisx, thisy, interp)

    maxval = np.max(energy_sum)
    maxx, maxy = np.nonzero(energy_sum == maxval)

    xoff = xoffs[maxx[0]]
    yoff = yoffs[maxy[0]]

    return xoff, yoff


def compute_energy_sum(x, y, interp):
    """
    Compute the sum of interpolated energies for positions x, y
    in image represented by object interp.
    """
    E = 0
    for i in xrange(len(x)):
        E += interp(x[i], y[i])
    return E


def get_arc(mom):
    # circle center: center of circle which contains the arc.
    phi2 = np.abs(mom.phi) / 2
    circle_center_rot = [0, -mom.R * np.sin(phi2) / phi2]

    # arc points
    phi0 = np.pi / 2 - phi2
    phi1 = np.pi / 2 + phi2
    phi = np.linspace(phi0, phi1, 1000)
    arc_rot = np.array([mom.R * np.cos(phi) + circle_center_rot[0],
                        mom.R * np.sin(phi) + circle_center_rot[1]])
    return arc_rot


def plot_arrow(start_xy, direction_rad, length=5, headlength=1,
               flipxy=True, color='c', lw=2):
    headangle = 45 * np.pi / 180.0
    leftangle = direction_rad + headangle
    rightangle = direction_rad - headangle
    head_dxy = np.array([length * np.cos(direction_rad),
                         length * np.sin(direction_rad)])
    dx = np.array([
        0, head_dxy[0],
        head_dxy[0] - headlength * np.cos(leftangle),
        head_dxy[0],
        head_dxy[0] - headlength * np.cos(rightangle)])
    dy = np.array([
        0, head_dxy[1],
        head_dxy[1] - headlength * np.sin(leftangle),
        head_dxy[1],
        head_dxy[1] - headlength * np.sin(rightangle)])
    x = start_xy[0] + dx
    y = start_xy[1] + dy
    if flipxy:
        plt.plot(y, x, color, lw=lw)
    else:
        plt.plot(x, y, color, lw=lw)


def get_arc2(mom):
    """
    get arc and start point for plot. (arc_full and EP_full)
    """

    arc_rot = get_arc(mom)
    # moments-calculated entry point EP a.k.a. x0
    EP_central = mom.x0
    # transform into full image frame
    arc_full = mom.rotated_to_full(arc_rot)
    EP_full = mom.central_to_full(EP_central)

    return arc_full, EP_full


def plot_moments_track(mom, track, title=''):
    """
    Plot track for Don. Left pane shows full track, right pane is zoomed.
    On each there is the segment box, the arc, the computed arrow, and the
    Monte Carlo arrow.
    """

    # use get_arc() and plot_arrow() to get all the coordinates we need.

    # geant4 entry point
    g4x, g4y = get_image_xy(track)
    g4x0 = np.array([g4x[0], g4y[0]])

    # geant4 direction (converted to radians)
    alpha_true = track.g4track.alpha_deg * np.pi / 180.0

    good_moments = False
    if mom is not None:
        if not np.isnan(mom.rotation_angle):
            good_moments = True
    if good_moments:
        arc_full, EP_full = get_arc2(mom)
        # moments-calculated direction alpha (radians)
        #   (in segment and full frames)
        alpha = mom.alpha

    # create figure
    f = plt.figure(figsize=(16, 8))

    # left plot: full image
    plt.subplot(1, 2, 1)
    cmap = get_colormap()
    plt.imshow(track.image, cmap=cmap, aspect='equal',
               interpolation='none', origin='lower')
    if good_moments:
        plt.plot(mom.box_y, mom.box_x, 'c')
        plt.plot(arc_full[1, :], arc_full[0, :], 'c', lw=2.5)
        plot_arrow(EP_full, alpha, color='c')
    plot_arrow(g4x0, alpha_true, color='g')
    plt.xlim((0, track.image.shape[1] - 1))
    plt.ylim((0, track.image.shape[0] - 1))
    plt.xlabel('y [pixels]')
    plt.ylabel('x [pixels]')
    plt.colorbar()
    plt.title(title)

    # right plot: zoomed
    if good_moments:
        plt.subplot(1, 2, 2)
        plt.imshow(mom.original_image_kev, cmap=cmap, aspect='equal',
                   interpolation='none', origin='lower')
        plt.plot(mom.box_y, mom.box_x, 'c')
        plt.plot(arc_full[1, :], arc_full[0, :], 'c', lw=2.5)
        plot_arrow(EP_full, alpha, color='c')
        plot_arrow(g4x0, alpha_true, color='g')

        xoff = mom.end_segment_offsets[1]
        yoff = mom.end_segment_offsets[0]
        plt.xlim((xoff - 1, xoff + mom.end_segment_image.shape[1] + 1))
        plt.ylim((yoff - 1, yoff + mom.end_segment_image.shape[0] + 1))
        plt.xlabel('y [pixels]')
        plt.ylabel('x [pixels]')
        plt.colorbar()
        plt.title(title)

    return f


def plot_arc(center, radius, phi0, phi1, flipxy=True):
    phi = np.linspace(phi0, phi1, 1000)
    x = center[0] + radius * np.cos(phi)
    y = center[1] + radius * np.sin(phi)
    if flipxy:
        plt.plot(y, x, 'c', lw=4)
    else:
        plt.plot(x, y, 'c', lw=4)


def get_colormap():
    """
    Return 'hot log' colormap, our standard colormap for electron tracks.
    """

    # like "hot" colormap, but more logarithmic rather than linear.
    # copy of colormap 'cmaphotlog' from MATLAB.
    cdict = {'red':     ((0.0,   0.0, 0.0),
                         (0.125, 1.0, 1.0),
                         (1.0,   1.0, 1.0)),
             'green':   ((0.0,   0.0, 0.0),
                         (0.125, 0.0, 0.0),
                         (0.375, 1.0, 1.0),
                         (1.0,   1.0, 1.0)),
             'blue':    ((0.0,   0.0, 0.0),
                         (0.375, 0.0, 0.0),
                         (1.0,   1.0, 1.0))
             }
    cmap_hot_log = matplotlib.colors.LinearSegmentedColormap('hotlog', cdict)

    return cmap_hot_log


class G4TrackTooBigError(Exception):
    pass


if False:
    pdb.set_trace()
    pass
