#!/usr/bin/python

import numpy as np
import skimage.morphology as morph

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

    def get_segment_initial_values(self):
        """
        Get start_coordinates, end_coordinates, and rough_est for segmenting.
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

    def get_segment_box(self):
        """
        Get the x,y coordinates of the box containing the initial segment.
        """

        segwid = 10   # pixels
        seglen = 11  # pixels

        mod = self.rough_est % (np.pi / 2)
        if mod < np.pi / 6 or mod > np.pi / 3:
            # close enough to orthogonal
            general_dir = np.round(self.rough_est / (np.pi / 2)) * (np.pi / 2)
            self.is45 = False
        else:
            # use a diagonal box (aligned to 45 degrees)
            general_dir = np.round(self.rough_est / (np.pi / 4)) * (np.pi / 4)
            self.is45 = True

        # make box and rotate
        box_dx = np.array([0, -seglen, -seglen, 0, 0])
        box_dy = np.array(
            [-segwid / 2, -segwid / 2, segwid / 2, segwid / 2, -segwid / 2])
        box_dx_rot = (np.round(box_dx * np.cos(general_dir)) -
                      np.round(box_dy * np.sin(general_dir))).astype(int)
        box_dy_rot = (np.round(box_dx * np.sin(general_dir)) +
                      np.round(box_dy * np.cos(general_dir))).astype(int)
        self.box_x = self.end_coordinates[0] + box_dx_rot
        self.box_y = self.end_coordinates[1] + box_dy_rot

    def check_segment_box(self):
        """
        Check whether the box intersects too many hot pixels.
        That would indicate that we should draw a new box.
        """

        problem_length = 60     # microns

        # xmesh, ymesh get used in get_pixlist, also. so save into self.
        img_shape = self.original_image_kev.shape
        self.xmesh, self.ymesh = np.meshgrid(
            range(img_shape[0]), range(img_shape[1]), indexing='ij')
        # get the pixels along the line segment that passes through the track,
        #   by walking along from one endpoint toward the other.
        xcheck = [self.box_x[-2]]
        ycheck = [self.box_y[-2]]
        dx = np.sign(self.box_x[-1] - self.box_x[-2])
        dy = np.sign(self.box_y[-1] - self.box_y[-2])
        while xcheck[-1] != self.box_x[-1] or ycheck[-1] != self.box_y[-1]:
            xcheck.append(xcheck[-1] + dx)
            ycheck.append(ycheck[-1] + dy)
        xcheck.append(self.box_x[-1])
        ycheck.append(self.box_y[-1])
        xcheck = np.array(xcheck)
        ycheck = np.array(ycheck)
        lgbad = ((xcheck < 0) | (xcheck >= self.original_image_kev.shape[0]) |
                 (ycheck < 0) | (ycheck >= self.original_image_kev.shape[1]))
        xcheck = xcheck[np.logical_not(lgbad)]
        ycheck = ycheck[np.logical_not(lgbad)]

        # threshold from HybridTrack options
        low_threshold_kev = self.options.low_threshold_kev

        # see what pixels are over the threshold.
        over_thresh = np.array(
            [self.original_image_kev[xcheck[i], ycheck[i]] > low_threshold_kev
             for i in xrange(len(xcheck))])
        # in order to avoid counting pixels from a separate segment,
        #   start from end_coordinates and count outward until you hit a 0.
        over_thresh_pix = 1
        start_ind = np.nonzero(
            (xcheck == self.end_coordinates[0]) &
            (ycheck == self.end_coordinates[1]))[0][0]
        # +dx, +dy side (start_ind+1 --> end):
        for i in xrange(start_ind + 1, len(xcheck), 1):
            if over_thresh[i]:
                over_thresh_pix += 1
            else:
                break
        # -dx, -dy side (start_ind-1 --> 0):
        for i in xrange(start_ind - 1, -1, -1):
            if over_thresh[i]:
                over_thresh_pix += 1
            else:
                break

        over_thresh_length = (
            over_thresh_pix *
            self.options.pixel_size_um * np.sqrt(dx**2 + dy**2))

        if over_thresh_length > problem_length:
            # have we done this too much already?
            if self.options.ridge_starting_distance_from_track_end_um < 30:
                raise CheckSegmentBoxError("Couldn't get a clean end segment")
                return None
            # try again, with a shorter track segment
            self.options.ridge_starting_distance_from_track_end_um -= 10.5
            # now, repeat what we've done so far
            hybridtrack.choose_initial_end(
                self.original_image_kev, self.options, self.info)
            self.get_segment_initial_values()
            self.get_segment_box()
            self.check_segment_box()
            # recurse until ridge_starting_dist... < 30

    def get_pixlist(self):
        """
        get list of pixels that are inside the box (and within image bounds)
        """

        # logical array representing pixels in the segment
        if not self.is45:
            segment_lg = ((self.xmesh >= np.min(self.box_x)) &
                          (self.xmesh <= np.max(self.box_x)) &
                          (self.ymesh >= np.min(self.box_y)) &
                          (self.ymesh <= np.max(self.box_y)))
        else:
            # need to compose the lines which form bounding box
            pairs = ((0, 1), (1, 2), (2, 3), (3, 0))
            m = np.zeros(4)
            b = np.zeros(4)
            for i in xrange(len(pairs)):
                # generate the m, b for y = mx+b for line connecting this
                #   pair of points
                m[i] = ((self.box_y[pairs[i][1]] - self.box_y[pairs[i][0]]) /
                        (self.box_x[pairs[i][1]] - self.box_x[pairs[i][0]]))
                b[i] = self.box_y[pairs[i][0]] - m[i] * self.box_x[pairs[i][0]]
            # m should be alternating sign... (this is for testing)
            assert m[0] * m[1] == -1
            assert m[1] * m[2] == -1
            assert m[2] * m[3] == -1
            assert m[3] * m[0] == -1
            min_ind = np.zeros(2)
            max_ind = np.zeros(2)
            if b[0] < b[2]:
                min_ind[0] = 0
                max_ind[0] = 2
            else:
                min_ind[0] = 2
                max_ind[0] = 0
            if b[1] < b[3]:
                min_ind[1] = 1
                max_ind[1] = 3
            else:
                min_ind[1] = 3
                max_ind[1] = 1
            segment_lg = (
                (self.ymesh >= m[min_ind[0]] * self.xmesh + b[min_ind[0]]) &
                (self.ymesh >= m[min_ind[1]] * self.xmesh + b[min_ind[1]]) &
                (self.ymesh <= m[max_ind[0]] * self.xmesh + b[max_ind[0]]) &
                (self.ymesh <= m[max_ind[1]] * self.xmesh + b[max_ind[1]]))

        xpix = self.xmesh[segment_lg]
        ypix = self.ymesh[segment_lg]

        self.xpix = xpix
        self.ypix = ypix

    def get_segment_image(self):
        """
        Produce the actual segment image using xpix and ypix
        """

        # initialize segment image
        min_x = np.min(self.xpix)
        max_x = np.max(self.xpix)
        min_y = np.min(self.ypix)
        max_y = np.max(self.ypix)
        seg_img = np.zeros((max_x - min_x + 1, max_y - min_y + 1))

        # fill segment image
        for i in xrange(len(self.xpix)):
            xi = self.xpix[i] - min_x
            yi = self.ypix[i] - min_y
            seg_img[xi, yi] = self.original_image_kev[
                self.xpix[i], self.ypix[i]]

        self.end_segment_image = seg_img
        self.end_segment_offsets = np.array([min_x, min_y])

    def separate_segments(self):
        """
        Perform image segmentation on the "segment image", and remove any
        segments that aren't the right part of the track.
        """

        # binary image
        binary_segment_image = (
            self.end_segment_image > self.options.low_threshold_kev)
        # segmentation: labeled regions, 8-connectivity
        labels = morph.label(binary_segment_image, connectivity=2)
        x1 = self.end_coordinates[0] - self.end_segment_offsets[0]
        y1 = self.end_coordinates[1] - self.end_segment_offsets[1]
        x2 = self.start_coordinates[0] - self.end_segment_offsets[0]
        y2 = self.start_coordinates[1] - self.end_segment_offsets[1]
        chosen_label = labels[x1, y1]
        if labels[x2, y2] != chosen_label:
            # this happens with 4-connectivity. need to use 8-connectivity
            raise RuntimeError('What the heck happened?')
        binary_again = (labels == chosen_label)
        # dilate this region, in order to capture information below threshold
        #  (it won't include the other regions, because there must be a gap
        #   between)
        pix_to_keep = morph.binary_dilation(binary_again)
        self.end_segment_image[np.logical_not(pix_to_keep)] = 0

    def segment_initial_end(self):
        """
        Get the image segment to use for moments, and the rough direction
        estimate.

        Calls get_segment_box(), get_pixlist(), get_segment_image().
        """

        self.get_segment_initial_values()
        self.get_segment_box()
        self.check_segment_box()
        self.get_pixlist()
        self.get_segment_image()
        self.separate_segments()

        def end_segment_coords_to_full_image_coords(xy):
            """
            Convert x,y from the coordinate frame of the end segment image
            to the coordinate frame of the full image
            """
            x, y = xy_split(xy)
            return np.array([x + self.end_segment_offsets[0],
                             y + self.end_segment_offsets[1]])

        self.segment_to_full = end_segment_coords_to_full_image_coords

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
            x, y = xy_split(xy)
            return np.array([x + self.xoffset, y + self.yoffset])

        self.central_to_segment = central_coords_to_end_segment_coords

        def central_coords_to_full_image_coords(xy):
            """
            Convert x,y from the coordinate frame of the central moments
            to the coordinate frame of the end segment image
            """
            return self.segment_to_full(self.central_to_segment(xy))

        self.central_to_full = central_coords_to_full_image_coords

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
            x, y = xy_split(xy)
            t = self.rotation_angle
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


def xy_split(xy):
    xy = np.array(xy)
    if xy.ndim == 1:
        x = xy[0]
        y = xy[1]
    elif xy.ndim == 2 and xy.shape[0] == 2:
        x = xy[0, :].flatten()
        y = xy[1, :].flatten()
    else:
        x = xy[:, 0].flatten()
        y = xy[:, 1].flatten()
    return x, y


class MomentsError(Exception):
    pass


class CheckSegmentBoxError(MomentsError):
    pass
