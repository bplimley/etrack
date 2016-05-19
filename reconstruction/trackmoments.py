#!/usr/bin/python

import numpy as np

from etrack.reconstruction import hybridtrack

# Don's summary from November 2015:
#  (0. separate initial segment)
#   1. compute moments
#   2a. determine center of image
#   2b. recalculate moments relative to center
#   3a. rotate image to principal axes of distribution
#      (positive x-axis in direction of motion)
#   4. find the arc parameters in the principle axis frame
#   5. find angle of entry, point of entry, arc length, and rate of
#      charge deposition from arc parameters


class MomentsReconstruction(object):

    def __init__(self, original_image_kev, pixel_size_um=10.5):
        """
        Init: Load options only
        """

        self.original_image_kev = original_image_kev
        self.options = hybridtrack.ReconstructionOptions(pixel_size_um)
        self.info = hybridtrack.ReconstructionInfo()

    def reconstruct(self):

        hybridtrack.choose_initial_end(
            self.original_image_kev, self.options, self.info)

        self.segment_initial_end()

        self.compute_first_moments()
        self.compute_central_moments()

        self.compute_direction()

    def compute_first_moments(self):

        self.clist0 = CoordinatesList.from_image(self.end_segment_image)

        self.first_moments = get_moments(self.clist0, maxmoment=1)

    def compute_central_moments(self):
        self.clist1 = CoordinatesList.from_clist(
            self.clist0,
            xoffset=self.first_moments[1, 0] / self.first_moments[0, 0],
            yoffset=self.first_moments[0, 1] / self.first_moments[0, 0])

        self.central_moments = get_moments(self.clist1, maxmoment=3)


class CoordinatesList(object):

    def __init__(self, xlist, ylist, Elist):
        self.x = xlist
        self.y = ylist
        self.E = Elist

    @classmethod
    def from_image(cls, image):
        xi = np.array(range(image.shape[0]))
        yi = np.array(range(image.shape[1]))
        xx, yy = np.meshgrid(xi, yi, indexing='ij')

        coordlist = cls(xx.flatten(), yy.flatten(), image.flatten())
        return coordlist

    @classmethod
    def from_clist(cls, clist, xoffset=0.0, yoffset=0.0):
        xlist = clist.x - xoffset
        ylist = clist.y - yoffset
        coordlist = cls(xlist, ylist, clist.E)
        return coordlist


def get_moment(clist, r, s):
    return np.sum(clist.E * clist.x**r * clist.y**s)


def get_moments(clist, maxmoment=1):
    """
    Get moments up to order maxmoment.

    E.g.: maxmoment = 2, get T00; T10, T01; T20, T11, T02

    Output:
      array
      T[0, 0]
      T[1, 0]
      T[0, 1]
      T[i, j] is empty for i + j > maxmoment
    """
    arraymax = maxmoment + 1    # because moments start at 0
    T = np.empty((arraymax, arraymax))
    for i in xrange(arraymax):
        for j in xrange(arraymax):
            if i + j < arraymax:
                T[i, j] = get_moment(clist.x, clist.y, clist.E, i, j)

    return T


def segment_initial_end(image_kev, options, info):
    """
    """

    pass


def compute_moments(options, info):
    """
    """

    pass


def compute_direction(options, info):
    """
    """

    pass
