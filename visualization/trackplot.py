import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.patches as patches
import matplotlib

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


def plot_moments_segment(full_image, box_x, box_y):
    """
    Plot a track image and draw a box where trackmoments wants to work.

    box is [min_x, max_x, min_y, max_y]
    """

    f = plt.figure()
    ax, im = plot_track_image(full_image)

    plt.plot(box_y, box_x, 'c', axes=ax)

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
        center_in_image_frame = center_in_segment_frame + mom.end_segment_offsets

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


if False:
    pdb.set_trace()
    pass
