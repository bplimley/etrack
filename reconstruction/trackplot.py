import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors

# get RidgePoint and Cut classes
import hybridtrack


def plot_track_image(img):
    """
    """

    cmap = get_colormap()
    ax = plt.axes()
    im = plt.imshow(img, cmap=cmap, aspect='equal', interpolation='none',
        origin='lower')

    return ax, im


def plot_ridgepoints(ax, ridge_points, fmtstring='c.', **kwargs):
    """
    """

    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()

    if len(ridge_points) > 1:
        coordinates = np.array([r.coordinates_pix for r in ridge_points])
    else:
        coordinates = ridge_points.coordinates_pix

    pts = plt.plot(coordinates[:,1],
                  coordinates[:,0],
                  fmtstring,
                  axes=ax,
                  **kwargs)

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
    c = plt.plot(coordinates[:,1],
                 coordinates[:,0],
                 fmtstring,
                 axes=ax,
                 **kwargs)

    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)

    return c


def get_colormap():
    """Return 'hot log' colormap, our standard colormap for electron tracks.
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
