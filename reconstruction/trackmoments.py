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

        self.compute_moments()

        self.compute_direction()

    def compute_first_moments(self):
        clist = CoordinatesList.from_image(self.end_segment_image)
        T = np.empty((2, 2))
        for i in xrange(2):
            for j in xrange(2):
                if i + j < 2:
                    T[i, j] = get_moment(clist.x, clist.y, clist.E, i, j)
        self.first_moments = T


class CoordinatesList(object):

    def __init__(self, xlist, ylist, Elist):
        self.x = xlist
        self.y = ylist
        self.E = Elist

    @classmethod
    def from_image(cls, image, xoffset=0.0, yoffset=0.0):
        xi = np.array(range(image.shape[0]))
        yi = np.array(range(image.shape[1]))
        xi -= xoffset
        yi -= yoffset
        xx, yy = np.meshgrid(xi, yi, indexing='ij')

        coordlist = cls(xx.flatten(), yy.flatten(), image.flatten())
        return coordlist


def get_moment(xlist, ylist, elist, r, s):
    return np.sum(elist * xlist**r * ylist**s)


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
