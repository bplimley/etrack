import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors

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


def plot_moments_segment(full_image, box):
    """
    Plot a track image and draw a box where trackmoments wants to work.

    box is [min_x, max_x, min_y, max_y]
    """

    ax, im = plot_track_image(full_image)

    xc = np.array([box[0], box[0], box[1], box[1], box[0]])
    yc = np.array([box[2], box[3], box[3], box[2], box[2]])
    plt.plot(xc, yc, fmstring='c', axes=ax)


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
