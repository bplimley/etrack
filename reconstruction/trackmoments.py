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

    def __init__(self, original_image_kev, pixel_size_um=10.5,
                 starting_distance=63):
        """
        Init: Load options only
        """

        self.original_image_kev = original_image_kev
        self.options = hybridtrack.ReconstructionOptions(pixel_size_um)
        # increase walking distance from 4 pixels to 6 pixels - for now
        # hybridtrack default: 40 um
        # initial testing before 6/21/16: 63 um
        self.options.ridge_starting_distance_from_track_end_um = starting_distance

        self.info = hybridtrack.ReconstructionInfo()

    def reconstruct(self):

        hybridtrack.choose_initial_end(
            self.original_image_kev, self.options, self.info)

        self.segment_initial_end()
        # get a sub-image containing the initial end
        # also need a rough estimate of the electron direction (from thinned)

        # 1.
        self.get_coordlist()
        self.compute_first_moments()
        # 2ab.
        self.compute_central_moments()
        # 3ab.
        self.compute_optimal_rotation_angle()
        #  4.
        self.compute_arc_parameters()

        self.compute_direction()

        self.compute_pathology()

    @classmethod
    def reconstruct_test(cls, end_segment_image, rough_est):
        """
        Run the moments on a (handpicked) track section.
        """

        mom = cls(None)
        mom.end_segment_image = end_segment_image
        mom.rough_est = rough_est

        mom.get_coordlist()
        mom.compute_first_moments()
        mom.compute_central_moments()
        mom.compute_optimal_rotation_angle()
        mom.compute_arc_parameters()
        mom.compute_direction()

        return mom

    @classmethod
    def reconstruct_arc(cls, clist, rough_est):
        """
        Run the moments on a CoordinatesList object, e.g. from generate_arc().

        Need to also provide a rough_est
        """

        mom = cls(None)
        mom.clist0 = clist
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

        orig = self.original_image_kev

        # Segment the image

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
                raise RuntimeError(
                    'square, but general_dir = {}'.format(general_dir))
            # don't go outside of image
            min_x = np.maximum(min_x, 0)
            max_x = np.minimum(max_x, orig.shape[0])
            min_y = np.maximum(min_y, 0)
            max_y = np.minimum(max_y, orig.shape[1])
            self.box_x = np.array([
                min_x, min_x, max_x, max_x, min_x])
            self.box_y = np.array([
                min_y, max_y, max_y, min_y, min_y])
            # draw box
            self.end_segment_image = orig[min_x:max_x, min_y:max_y]

        else:
            # close to 45 degrees
            is45 = True

            diag_hw = int(np.round(float(segment_width / 2) / np.sqrt(2)))
            diag_len = segment_length
            # don't divide by sqrt(2) because the choose_initial_end doesn't
            #   distinguish between diagonal steps and orthogonal steps.
            img_shape = orig.shape
            xmesh, ymesh = np.meshgrid(
                range(img_shape[0]), range(img_shape[1]), indexing='ij')

            # for whichever direction, start from end_coordinates and draw box

            # general_dir is from start toward end
            general_dir = np.round(self.rough_est / (np.pi / 4)) * np.pi / 4
            # bdir is from end toward start
            bdir = general_dir + np.pi

            xsign = np.sign(np.cos(bdir))
            ysign = np.sign(np.sin(bdir))

            x0 = self.end_coordinates[0]
            y0 = self.end_coordinates[1]

            # draw the box
            self.box_x = np.array([
                x0 - xsign * diag_hw,
                x0 - xsign * diag_hw + xsign * diag_len,
                x0 + xsign * diag_hw + xsign * diag_len,
                x0 + xsign * diag_hw,
                x0 - xsign * diag_hw,
            ])
            self.box_y = np.array([
                y0 + ysign * diag_hw,
                y0 + ysign * diag_hw + ysign * diag_len,
                y0 - ysign * diag_hw + ysign * diag_len,
                y0 - ysign * diag_hw,
                y0 + ysign * diag_hw,
            ])

            # make a list of pixels in the box

            base_xpix, base_ypix = self.get_base_diagonal_pixlist(
                diag_hw, diag_len)
            rot = bdir - np.pi / 4      # angle of rotation
            # round to make sure they are integers
            xpix = x0 + np.round(
                base_xpix * np.cos(rot) - base_ypix * np.sin(rot))
            ypix = y0 + np.round(
                base_xpix * np.sin(rot) + base_ypix * np.cos(rot))
            # exclude out-of-bounds points
            shape = orig.shape
            out_of_bounds = ((xpix < 0) | (ypix < 0) |
                             (xpix >= shape[0]) | (ypix >= shape[1]))
            xpix = xpix[np.logical_not(out_of_bounds)]
            ypix = ypix[np.logical_not(out_of_bounds)]

            # initialize segment image
            min_x = np.min(xpix)
            max_x = np.max(xpix)
            min_y = np.min(ypix)
            max_y = np.max(ypix)
            seg_img = np.zeros((max_x - min_x + 1, max_y - min_y + 1))

            # fill segment image
            for i in xrange(len(xpix)):
                seg_img[xpix[i] - min_x, ypix[i] - min_y] = orig[
                    xpix[i], ypix[i]]

            self.end_segment_image = seg_img

            self.end_segment_offsets = np.array([min_x, min_y])

            def end_segment_coords_to_full_image_coords(xy):
                """
                Convert x,y from the coordinate frame of the end segment image
                to the coordinate frame of the full image
                """
                return np.array(xy) + self.end_segment_offsets

            self.segment_to_full = end_segment_coords_to_full_image_coords

            if False and is45:
                # debug
                print('min_x, max_x = ({}, {})'.format(min_x, max_x))
                print('box_x = {}'.format(self.box_x))
                print('min_y, max_y = ({}, {})'.format(min_y, max_y))
                print('box_y = {}'.format(self.box_y))

                # import ipdb as pdb; pdb.set_trace()

    @classmethod
    def get_base_diagonal_pixlist(cls, diag_hw, diag_len):
        """
        Get the diagonal pixel list for 45 degrees.
        (To be rotated to other angles)
        """

        xlist = []
        ylist = []

        # major diagonals
        xy0 = np.array([0, 0])
        xy1 = np.array([diag_len, diag_len])
        for i in range(-diag_hw, diag_hw):
            offset = np.array([i, -i])
            xt, yt = cls.list_diagonal_pixels(xy0 + offset, xy1 + offset)
            xlist += xt
            ylist += yt

        # minor diagonals
        xy0 = np.array([0, 1])
        xy1 = np.array([diag_len - 1, diag_len])
        for i in range(-diag_hw + 1, diag_hw):
            offset = np.array([i, -i])
            xt, yt = cls.list_diagonal_pixels(xy0 + offset, xy1 + offset)
            xlist += xt
            ylist += yt

        return np.array(xlist), np.array(ylist)

    @classmethod
    def list_diagonal_pixels(cls, xy0, xy1):
        """
        Return xlist, ylist, which list all the pixels on the 45-degree
        diagonal between xy0 and xy1.
        """

        dxy = [np.sign(xy1[0] - xy0[0]), np.sign(xy1[1] - xy0[1])]
        xlist = range(xy0[0], xy1[0], dxy[0])
        ylist = range(xy0[1], xy1[1], dxy[1])

        return xlist, ylist

    def get_coordlist(self):
        self.clist0 = CoordinatesList.from_image(self.end_segment_image)

    def compute_first_moments(self):
        self.first_moments = get_moments(self.clist0, maxmoment=1)

    def compute_central_moments(self):
        self.xoffset = self.first_moments[1, 0] / self.first_moments[0, 0]
        self.yoffset = self.first_moments[0, 1] / self.first_moments[0, 0]
        self.clist1 = CoordinatesList.from_clist(
            self.clist0, xoffset=self.xoffset, yoffset=self.yoffset)

        self.central_moments = get_moments(self.clist1, maxmoment=3)

        def central_coords_to_end_segment_coords(xy):
            """
            Convert x,y from the coordinate frame of the central moments
            to the coordinate frame of the end segment image
            """
            return xy + np.array([self.xoffset, self.yoffset])

        self.central_to_segment = central_coords_to_end_segment_coords

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
        dtheta = theta - self.rough_est
        dtheta[dtheta > np.pi] -= 2 * np.pi
        dtheta[dtheta < -np.pi] += 2 * np.pi
        condB = np.abs(dtheta) <= np.pi / 2
        # choose
        cond_both = condA & condB
        if not np.any(cond_both):
            raise MomentsError('Rotation quadrant conditions not met')
        elif np.sum(cond_both) > 1:
            pass
            # should throw out this event.
            # raise MomentsError(
            #     'Rotation quadrant conditions met more than once')
        chosen_ind = np.nonzero(cond_both)[0][0]    # 1st dim, 1st entry
        self.rotation_angle = theta[chosen_ind]
        self.clist2 = rotated_clists[chosen_ind]
        self.rotated_moments = rotated_moments[chosen_ind]

        def rotated_coords_to_central_coords(xy):
            """
            Convert x,y from the coordinate frame of the rotated moments
            to the coordinate frame of the central moments
            """
            xy = np.array(xy)
            t = self.rotation_angle
            if xy.ndim == 1:
                x = xy[0]
                y = xy[1]
            elif xy.ndim == 2 and xy.shape[0] == 2:
                x = xy[0, :].flatten()
                y = xy[1, :].flatten()
            else:
                x = xy[:, 0].flatten()
                y = xy[:, 1].flatten()
            # rotate "forward" because CoordList rotates "backward"
            x1 = x * np.cos(t) - y * np.sin(t)
            y1 = x * np.sin(t) + y * np.cos(t)
            return np.array([x1, y1])

        self.rotated_to_central = rotated_coords_to_central_coords

        def rotated_coords_to_end_segment_coords(xy):
            """
            Convert x,y from the coordinate frame of the rotated moments
            to the coordinate frame of the end segment image.
            """
            return self.central_to_segment(self.rotated_to_central(xy))

        self.rotated_to_segment = rotated_coords_to_end_segment_coords

        def rotated_coords_to_full_image_coords(xy):
            """
            Convert x,y from the coordinate frame of the rotated moments
            to the coordinate frame of the full image.
            """
            return self.segment_to_full(self.rotated_to_segment(xy))

        self.rotated_to_full = rotated_coords_to_full_image_coords

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

    def compute_pathology(self):
        # Test 1: The total arc-length should be longer than 3(?) pixels
        self.arclength = self.Rphi
        # Test 2: Radius should be much greater than arc-length
        # self.phi
        # Test 3: Certain moments are expected to ... be significantly smaller
        self.pathology_ratio_3a = (
            self.rotated_moments[1, 2] / self.rotated_moments[2, 1])
        self.pathology_ratio_3b = (
            self.rotated_moments[3, 0] / self.rotated_moments[0, 3])


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
        # rotation, if any. *coordinate frame* is rotated (theta -> -theta)
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


def generate_arc(r=6, phi_d=90, center_angle_d=0,
                 n_pts=1000, blur_sigma=0, pixelize=False):
    """
    Generate a CoordinatesList for a perfect arc.

    Inputs:
      r: arc radius [pixels] (default 6)
      phi_d: total arc angular length [degrees] (default 90)
      center_angle_d: where the arc is centered at; i.e. rotation. [degrees]
          (default 0)
      n_pts: total number of points generated. Equally spaced.
          Each will signify 1 energy unit (default 1000)
      blur_sigma: The sigma of a 2D Gaussian for randomly sampling the position
          of each point. [pixels] (default 0, no blurring)
      pixelize: Whether to pixelize the image and resample from the center of
          each pixel. [boolean] (default False)

    Output:
      clist: a CoordinatesList object
      rough_est: the initial direction (on the low-angle end)
    """

    phi_d = float(phi_d)
    n_pts = int(n_pts)
    angular_interval_d = phi_d / (n_pts - 1)
    phi0 = center_angle_d - phi_d / 2
    phi1 = center_angle_d + phi_d / 2

    phi_all_d = np.arange(phi0, phi1 + angular_interval_d, angular_interval_d)
    phi_all_r = phi_all_d * np.pi / 180

    xlist = r * np.cos(phi_all_r)
    ylist = r * np.sin(phi_all_r)

    if blur_sigma > 0:
        xblur = np.random.randn(len(xlist)) * blur_sigma
        yblur = np.random.randn(len(xlist)) * blur_sigma
        xlist += xblur
        ylist += yblur

    if pixelize:
        # each pixel is centered on an integer value
        # round everything to nearest pixel
        xlist = np.round(xlist)
        ylist = np.round(ylist)

    clist = CoordinatesList(xlist, ylist, np.ones_like(xlist))

    # direction is basically (dx, dy) which is phi0 + 90 degrees
    rough_est = (phi0 + 90) * np.pi / 180

    return clist, rough_est


class MomentsError(Exception):
    pass
