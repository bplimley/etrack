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
#   3b. recalculate moments in rotated frame
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
        # increase walking distance from 4 pixels to 6 pixels - for now
        self.options.ridge_starting_distance_from_track_end_um = 63

        self.info = hybridtrack.ReconstructionInfo()

    def reconstruct(self):

        hybridtrack.choose_initial_end(
            self.original_image_kev, self.options, self.info)

        self.segment_initial_end()
        # get a sub-image containing the initial end
        # also need a rough estimate of the electron direction (from thinned)

        # 1.
        self.compute_first_moments()
        # 2ab.
        self.compute_central_moments()
        # 3ab.
        self.compute_optimal_rotation_angle()
        #  4.
        self.compute_arc_parameters()

        self.compute_direction()

    @classmethod
    def reconstruct_test(cls, end_segment_image, rough_est):
        """
        Run the moments on a (handpicked) track section.
        """

        mom = cls(None)
        mom.end_segment_image = end_segment_image
        mom.rough_est = rough_est

        mom.compute_first_moments()
        mom.compute_central_moments()
        mom.compute_optimal_rotation_angle()
        mom.compute_arc_parameters()
        mom.compute_direction()

        return mom

    def segment_initial_end(self):
        """
        Get the image segment to use for moments, and the rough direction
        estimate.
        """

        # copied from hybridtrack.get_starting_point()
        min_index = self.info.ends_energy.argmin()
        self.start_coordinates = self.info.ends_xy[min_index]
        # start_coordinates are the end (extremity) of the thinned track

        self.end_coordinates = self.info.start_coordinates
        # end_coordinates are after walking up the track 40 um

        # angle from start to end
        dcoord = self.end_coordinates - self.start_coordinates
        self.rough_est = np.arctan2(dcoord[1], dcoord[0])

        ## Segment the image
        segment_width = 10   # pixels
        segment_length = 9  # pixels


        mod = self.rough_est % (np.pi / 2)
        if mod < np.pi / 6 or mod > np.pi / 3:
            # close enough to orthogonal
            # round to nearest pi/4
            general_dir = np.round(self.rough_est / (np.pi / 2))

            # for whichever direction, start from end_coordinates and draw box
            if general_dir == 0 or general_dir == -4:
                # +x direction
                max_x = self.end_coordinates[0]
                min_x = max_x - segment_length
                min_y = self.end_coordinates[1] - segment_width / 2
                max_y = self.end_coordinates[1] + segment_width / 2
            elif general_dir == 1 or general_dir == -3:
                # +y direction
                max_y = self.end_coordinates[1]
                min_y = max_y - segment_length
                min_x = self.end_coordinates[0] - segment_width / 2
                max_x = self.end_coordinates[0] + segment_width / 2
            elif general_dir == 2 or general_dir == -2:
                # -x direction
                min_x = self.end_coordinates[0]
                max_x = min_x + segment_length
                min_y = self.end_coordinates[1] - segment_width / 2
                max_y = self.end_coordinates[1] + segment_width / 2
            elif general_dir == 3 or general_dir == -1:
                # -y direction
                min_y = self.end_coordinates[1]
                max_y = min_y + segment_length
                min_x = self.end_coordinates[0] - segment_width / 2
                max_x = self.end_coordinates[0] + segment_width / 2
            else:
                raise RuntimeError()
            # don't go outside of image
            min_x = np.maximum(min_x, 0)
            max_x = np.minimum(max_x, self.original_image_kev.shape[0])
            min_y = np.maximum(min_y, 0)
            max_y = np.minimum(max_y, self.original_image_kev.shape[1])
            self.box = np.array([min_x, max_x, min_y, max_y])
            # draw box
            self.end_segment_image = self.original_image_kev[
                min_x:max_x, min_y:max_y]

        else:
            # close to 45 degrees

            # do this later
            raise NotImplementedError

    def compute_first_moments(self):

        self.clist0 = CoordinatesList.from_image(self.end_segment_image)

        self.first_moments = get_moments(self.clist0, maxmoment=1)

    def compute_central_moments(self):
        self.clist1 = CoordinatesList.from_clist(
            self.clist0,
            xoffset=self.first_moments[1, 0] / self.first_moments[0, 0],
            yoffset=self.first_moments[0, 1] / self.first_moments[0, 0])

        self.central_moments = get_moments(self.clist1, maxmoment=3)

    def compute_optimal_rotation_angle(self):
        numerator = 2 * self.central_moments[1, 1]
        denominator = self.central_moments[2, 0] - self.central_moments[0, 2]
        theta0 = 0.5 * np.arctan(numerator / denominator)
        # four possible quadrants
        theta = np.array([0, np.pi / 2, np.pi, 3 * np.pi / 2]) + theta0
        rotated_clists = [
            CoordinatesList.from_clist(self.clist1, rotation_rad=t)
            for t in theta]
        rotated_moments = [
            get_moments(this_clist, maxmoment=3)
            for this_clist in rotated_clists]
        # condition A: x-axis is longer than y-axis
        condA = np.array([m[2, 0] - m[0, 2] > 0 for m in rotated_moments])
        # condition B: direction of rough estimate
        condB = np.abs(theta - self.rough_est) < np.pi / 2
        # choose
        cond_both = condA & condB
        if not np.any(cond_both):
            pass
            # raise an exception
        elif np.sum(cond_both) > 1:
            pass
            # raise a different exception
        chosen_ind = np.nonzero(cond_both)[0][0]    # 1st dim, 1st entry
        self.rotation_angle = theta[chosen_ind]
        self.clist2 = rotated_clists[chosen_ind]
        self.rotated_moments = rotated_moments[chosen_ind]

    def compute_arc_parameters(self):
        C_fit = -8.5467
        T = self.rotated_moments
        phi = C_fit * ((np.sqrt(T[0, 0]) * T[2, 1]) /
                       (T[2, 0] - T[0, 2]) ** 1.5)
        q1 = np.sqrt(2 - 2 * np.cos(phi) - phi * np.sin(phi))
        q2 = np.sqrt((T[2, 0] - T[0, 2]) / T[0, 0])
        q3 = np.sqrt(T[0, 0] ** 3 / (T[2, 0] - T[0, 2]))
        self.R = (phi / q1 * q2)
        self.Rphi = self.R * phi
        self.eta0 = (q1 / phi ** 2) * q3

        self.phi = phi
        self.arc_center = np.array([T[1, 0], T[0, 1]]) / T[0, 0]

    def compute_direction(self):
        self.alpha = self.phi / 2 + self.rotation_angle
        e_theta = np.array([np.cos(self.rotation_angle),
                            np.sin(self.rotation_angle)])
        e2 = np.array([np.cos(self.rotation_angle + np.pi / 2),
                       np.sin(self.rotation_angle + np.pi / 2)])
        q1 = self.R * np.sin(self.phi / 2) * e_theta
        # np.sinc is a NORMALIZED sinc function - sin(pi*x)/(pi*x)
        q2 = self.R * (np.cos(self.phi / 2) - np.sin(self.phi) / self.phi) * e2
        self.x0 = self.arc_center - q1 + q2


class CoordinatesList(object):

    def __init__(self, xlist, ylist, Elist):
        self.x = np.array(xlist)
        self.y = np.array(ylist)
        self.E = np.array(Elist)

    @classmethod
    def from_image(cls, image):
        xi = np.array(range(image.shape[0]))
        yi = np.array(range(image.shape[1]))
        xx, yy = np.meshgrid(xi, yi, indexing='ij')

        coordlist = cls(xx.flatten(), yy.flatten(), image.flatten())
        return coordlist

    @classmethod
    def from_clist(cls, clist, xoffset=0.0, yoffset=0.0, rotation_rad=0.0):
        # offset, if any
        xtemp = clist.x - xoffset
        ytemp = clist.y - yoffset
        # rotation, if any
        theta = rotation_rad
        xlist = xtemp * np.cos(theta) + ytemp * np.sin(theta)
        ylist = - xtemp * np.sin(theta) + ytemp * np.cos(theta)

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
                T[i, j] = get_moment(clist, i, j)

    return T


class MomentsError(Exception):
    pass
