#!/usr/bin/python

import numpy as np
import ipdb as pdb
import scipy.ndimage
import scipy.interpolate

import dedxref
import thinning
import trackplot


def reconstruct(track_object):
    img = track_object.image
    if track_object.pixel_size_um:
        pixel_size_um = track_object.pixel_size_um
    else:
        pixel_size_um = 10.5

    return reconstruct_from_image(img, pixel_size_um=pixel_size_um)


def reconstruct_from_image(original_image_kev, pixel_size_um=10.5):
    """
    Perform trajectory reconstruction on CCD electron track image.

    HybridTrack algorithm, UPDATED from MATLAB code, 2016-02-18.
    - first step alpha on [-180, 180]

    Inputs:
      pixel_size_um: pitch of pixels (default 10.5 um)

    Output:
      TBD.
    """

    # currently, only default options are supported, set in class def
    options = ReconstructionOptions(pixel_size_um)

    # add buffer of zeros around image
    track_energy_kev, prepared_image_kev = prepare_image(
        original_image_kev, options)

    # low threshold, thinning, identify ends
    info = ReconstructionInfo()
    choose_initial_end(prepared_image_kev, options, info)

    # ridge following
    ridge_follow(prepared_image_kev, options, info)

    compute_direction(track_energy_kev, options, info)

    info.prepared_image_kev = prepared_image_kev
    output = options   # temp
    return output, info
    # return set_output(prepared_image_kev, edge_segments, ridge, measurement,
    #                   options)


class ReconstructionOptions(object):
    """Reconstruction options for HybridTrack algorithm.

    Everything is set to defaults (based on pixel size) upon initialization.
    """

    def __init__(self, pixel_size_um):
        self.pixel_size_um = np.float(pixel_size_um)
        self.set_low_threshold()
        self.energy_kernel_radius_um = 25
        self.energy_kernel = construct_energy_kernel(
            self.energy_kernel_radius_um, self.pixel_size_um)
        self.set_ridge_options()
        self.set_measurement_options()

    def set_low_threshold(self):
        """Setting the low threshold for the binary image.
        """

        base_low_threshold_kev = 0.5
        scaling_factor = self.pixel_size_um**2 / 10.5**2
        self.low_threshold_kev = scaling_factor * base_low_threshold_kev
        self.pixel_area_scaling_factor = scaling_factor     # useful elsewhere

    def set_ridge_options(self):
        """Options for ridge-following.
        """

        self.ridge_starting_distance_from_track_end_um = 40

        # [larger = faster]
        self.position_step_size_pix = 0.25

        # [larger = faster]
        self.cut_sampling_interval_pix = 0.25

        # [smaller = faster]
        cut_total_length_um = 105
        self.cut_total_length_pix = (cut_total_length_um /
                                     self.pixel_size_um)

        # angle_increment_deg must be a factor of 45.
        # A smaller value might give a more accurate measurement of alpha.
        # [larger = faster]
        self.angle_increment_deg = 3
        self.angle_increment_rad = self.angle_increment_deg * np.pi / 180
        # angle_increment_deg is used as the base unit for all the other
        #   angular variables.
        # In other words, all variables ending in "indices" are in units of
        #   angle_increment_deg and can be multipled by it to get degrees.

        # search_angle_indices must be a multiple of 2
        # The maximum angular change in a single step is search_angle_indices/2
        # Therefore, larger search_angle allows tighter turns. But at 4 steps
        #   per pixel, search_angle does not need to be large. And regardless,
        #   search_angle_deg / 2 should be << 90, to minimize potential of
        #   walking onto an elbow or turning around.
        # [smaller = faster]
        # TODO: this could be smaller for smaller pixel sizes, depending on
        #   the position_step_size
        search_angle_deg = 48   # must be a multiple of 2*angle_increment_deg
        self.search_angle_ind = (search_angle_deg /
                                 self.angle_increment_deg)

        pi_deg = 180
        self.pi_ind = pi_deg / self.angle_increment_deg

        # all possible cut angles, for pre-calculating cuts
        #   *different from MATLAB code: starts with 0 instead of angle_incr
        self.cut_angle_rad = np.arange(0, 2 * np.pi, self.angle_increment_rad)
        self.cut_angle_deg = np.arange(0, 2 * pi_deg, self.angle_increment_deg)

        # other methods include ... what in python?
        self.cut_interpolation_method = 'linear'

        # Now, we define the x and y coordinates of the samples along a cut,
        #   and rotate to all possible angle increments.
        # The 0-degree cut is along the y-axis because the electron ridge is
        #   along the x-axis.
        cut0_y = np.arange(-self.cut_total_length_pix / 2,
                           self.cut_total_length_pix / 2 + 1e-3,  # with endpt
                           self.cut_sampling_interval_pix)
        cut0_x = np.zeros_like(cut0_y)
        cut0_coordinates = zip(tuple(cut0_x), tuple(cut0_y))
        # rotation matrices
        R = [np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]])
             for th in self.cut_angle_rad]
        self.cut_coordinates = [np.array(cut0_coordinates).dot(Ri) for Ri in R]
        # cut_distance_from_center is used for the width metric
        self.cut_distance_from_center_pix = np.abs(cut0_y)
        # cut_distance_coordinate is used for excluding points
        self.cut_distance_coordinate_pix = cut0_y

        # cut sample points with energy below cut_low_threshold, and beyond,
        #   are ignored.
        # scales with pixel area
        base_cut_low_threshold_kev = 0.05
        self.cut_low_threshold_kev = (base_cut_low_threshold_kev *
                                      self.pixel_area_scaling_factor)

        # when the ridge is below track_end_low_threshold, stop following it.
        # scales with pixel area
        base_track_end_low_threshold_kev = 0.1
        self.track_end_low_threshold_kev = (base_track_end_low_threshold_kev *
                                            self.pixel_area_scaling_factor)

        # distance threshold for catching an infinite loop
        self.infinite_loop_threshold_pix = self.position_step_size_pix / 2

    def set_measurement_options(self):
        """Options for calculating alpha and beta after the ridge following.
        """

        # min_width_measurement_len defined by diffusion, roughly
        min_width_measurement_len_um = 30
        preferred_width_measurement_len_pix = 2
        self.width_measurement_len_pix = np.max((
            preferred_width_measurement_len_pix,
            min_width_measurement_len_um / self.pixel_size_um))

        self.initial_beta_guess_deg = 45
        self.should_shorten_measurement_len = True
        self.measurement_func = np.median


class ReconstructionInfo(object):
    """Electron track info generated by HybridTrack algorithm.
    """

    def __init__(self):
        self.error = []
        self.threshold_used = []
        self.binary_image = []
        self.thinned_image = []
        self.ends_xy = []
        self.ends_energy = []
        # ...
        self.alpha = []
        self.beta = []


class ReconstructionOutput(object):
    """Electron track result from HybridTrack algorithm.
    """
    pass


class HybridTrackError(Exception):
    pass


class NoEndsFound(HybridTrackError):
    pass


class InfiniteLoop(HybridTrackError):
    pass


def construct_energy_kernel(radius_um, pixel_size_um):
    """Construct kernel for energy measurement.
    """

    radius_pix = radius_um / pixel_size_um
    ceil_radius_pix = int(np.ceil(radius_pix) - 1)
    coordinates = range(-ceil_radius_pix, ceil_radius_pix + 1)
    kernel = [
        [0 + (x**2 + y**2 < radius_pix**2) for x in coordinates]
        for y in coordinates]
    return np.array(kernel)


def prepare_image(image_kev, options):
    """Add a buffer of zeros around the original image.
    """

    # imageEdgeBuffer does not need to handle the cutTotalLength/2
    #   in any direction. it only needs to handle the ridge points going off
    #   the edge.

    # TODO: reduce buffer_width to... one pixel?

    buffer_width_um = 0.55 * options.cut_total_length_pix
    buffer_width_pix = np.ceil(buffer_width_um / options.pixel_size_um)
    orig_size = np.array(np.shape(image_kev))
    new_image_size = (orig_size + 2 * buffer_width_pix * np.ones(2))
    new_image_kev = np.zeros(new_image_size)

    indices_of_original = [
        [int(it) + int(buffer_width_pix) for it in range(orig_size[dim])]
        for dim in [0, 1]]
    # pdb.set_trace()
    new_image_kev[np.ix_(indices_of_original[0], indices_of_original[1])
                  ] = image_kev

    track_energy_kev = np.sum(np.sum(image_kev))

    return track_energy_kev, new_image_kev


def locate_all_ends(image_kev, options, info):
    """Apply threshold, perform thinning, identify ends.
    """
    ends_image = [False]
    current_threshold = options.low_threshold_kev
    connectivity = np.ones((3, 3))

    # normally this while loop only runs once.
    # if the track is a loop and so no ends are found, then increase the
    #   threshold until an end is found.
    # TODO: does this actually work?

    while np.all(np.logical_not(ends_image)) and (
            current_threshold <= 10 * options.low_threshold_kev):
        binary_image = image_kev > current_threshold
        thinned_image = thinning.thin(binary_image) + 0
        num_neighbors_image = scipy.ndimage.convolve(
            thinned_image, connectivity, mode='constant', cval=0.0) - 1
        num_neighbors_image = num_neighbors_image * thinned_image
        ends_image = num_neighbors_image == 1

        if np.all(np.logical_not(ends_image)):
            # increase threshold to attempt to break loop.
            current_threshold += options.low_threshold_kev

    info.binary_image = binary_image
    info.thinned_image = thinned_image

    if np.any(ends_image):
        ends_xy = np.where(ends_image)  # tuple of two lists (x and y)
        ends_xy = np.array(ends_xy).T   # list of coordinate pairs
        info.threshold_used = current_threshold
        info.ends_xy = ends_xy
    else:
        # still no ends.
        raise


def measure_energies(image_kev, options, info):
    """Compute energies using a kernel, for each end.
    """

    kernel = options.energy_kernel
    energy_convolved_image = scipy.ndimage.convolve(
        image_kev, kernel, mode='constant', cval=0.0)
    ends_energy = []
    for (x, y) in info.ends_xy:
        ends_energy.append(energy_convolved_image[x, y])

    info.ends_energy = np.array(ends_energy)


def get_starting_point(options, info):
    """
    Select minimum-energy end, walk up 40um, mark coordinates & direction.
    """
    min_index = info.ends_energy.argmin()
    n_steps_pix = int(np.ceil(
        options.ridge_starting_distance_from_track_end_um /
        options.pixel_size_um))
    xy = info.ends_xy[min_index]  # current position
    temp_image = info.thinned_image.copy()

    for i in range(n_steps_pix):
        temp_image[xy[0], xy[1]] = 0
        neighborhood = temp_image[xy[0]-1:xy[0]+2, xy[1]-1:xy[1]+2]
        n_neighbors = np.sum(neighborhood)      # center already removed
        if n_neighbors == 1:
            # step along the track
            step_xy = np.where(neighborhood)
            step_xy = np.array([step_xy[0][0], step_xy[1][0]])
            step_xy = step_xy - 1   # now represents a delta position
            xy = xy + step_xy
        elif n_neighbors > 1:
            # at an intersection. back up one.
            xy = xy - step_xy
            break
        elif n_neighbors == 0:
            # end of track
            break

    info.start_coordinates = xy
    info.start_direction_deg = (
        180/np.pi * np.arctan2(-step_xy[1], -step_xy[0]))
    if info.start_direction_deg < 0:
        info.start_direction_deg += 360
    info.start_direction_ind = (info.start_direction_deg /
                                options.angle_increment_deg)


def choose_initial_end(image_kev, options, info):
    """
    locate_all_ends
    measure_energies
    get_starting_point
    """

    # main
    locate_all_ends(image_kev, options, info)

    measure_energies(image_kev, options, info)
    get_starting_point(options, info)


def ridge_follow(image, options, info):
    """
    """

    size = np.shape(image)
    info.interp = scipy.interpolate.RectBivariateSpline(
        range(size[0]), range(size[1]), image, kx=1, ky=1)
    # RectBivariateSpline is 4x faster than interp2d!

    ridge = [RidgePoint(info.start_coordinates, info.start_direction_ind,
                        info=info, options=options)]

    while not ridge[-1].is_end:
        ridge.append(ridge[-1].step())
        if len(ridge) > 30:
            # until this length, not worth the time
            check_for_infinite_loop(ridge, options)

    ridge.pop()    # remove last item
    ridge.reverse()     # now 0 is the beginning of the track

    info.ridge = ridge


def check_for_infinite_loop(ridge, options):
    """See if the 2nd-to-last ridge point overlaps the previous ridge points.
    """

    if len(ridge) < 4:
        # too few points
        return None
    # The very last ridge point [-1] is not centroid-adjusted yet.
    # So use the one before that.
    coordinates = np.array(ridge[-2].coordinates_pix)
    # And skip the point neighboring it, because centroid-adjust could
    #   possibly bring it too close.
    previous = np.array([r.coordinates_pix for r in ridge[:-3]])
    distance_square = ((previous[:, 0] - coordinates[0])**2 +
                       (previous[:, 1] - coordinates[1])**2)
    if np.any(distance_square < options.infinite_loop_threshold_pix**2):
        raise InfiniteLoop
    else:
        return None


class RidgePoint(object):
    """
    """

    def __init__(self, xy, est_direction_ind,
                 previous=None, info=None, options=None):
        """Initialize a RidgePoint:

        1. save a reference to info and options
        2. put in estimated position and direction
        3. interpolate energy and check end condition
        4. placeholder variables
        """

        # store a reference to info and options objects for future reference
        if previous is None and (info is not None and options is not None):
            # case 1: beginning of ridge
            self.previous = None
            self.info = info
            self.options = options
        elif (info is None and options is None) and previous is not None:
            # case 2: copy info, options from previous ridge point
            self.info = previous.info
            self.options = previous.options
            self.previous = previous
        else:
            # case 3: problem
            raise ValueError('RidgePoint needs more information')

        # basic parameters needed to have a RidgePoint
        self.est_coordinates_pix = xy
        self.est_direction_ind = est_direction_ind

        # finally, get interpolated energy at this point and check threshold
        self.est_energy_kev = self.info.interp(self.est_coordinates_pix[0],
                                               self.est_coordinates_pix[1])
        self.is_end = (self.est_energy_kev <
                       self.options.track_end_low_threshold_kev)

        # placeholders
        self.coordinates_pix = []
        self.energy_kev = []
        self.cuts = []
        self.best_ind = []
        self.final_direction_deg = []
        self.dedx_kevum = []
        self.fwhm_um = []
        self.step_alpha_deg = []

    def step(self):
        """Figure out the next ridge point, and return it.
        """

        self.generate_all_cuts()
        self.choose_best_cut()
        self.adjust_to_centroid()
        self.measure_step_alpha()
        next_xy, next_dir_ind = self.estimate_next_step()

        self.next = RidgePoint(next_xy, next_dir_ind, self)

        return self.next

    def generate_all_cuts(self):
        """
        """

        angle_start = int(
            self.est_direction_ind - self.options.search_angle_ind / 2)
        angle_end = int(
            self.est_direction_ind + self.options.search_angle_ind / 2 + 1)
        these_indices = np.arange(angle_start, angle_end)
        # wrap around, 0 to 2pi
        these_indices[these_indices < 0] += 2*self.options.pi_ind
        these_indices[these_indices >= 2*self.options.pi_ind] -= (
            2*self.options.pi_ind)

        self.cuts = [Cut(self.info.interp, self.options,
                     self.est_coordinates_pix, angle_ind)
                     for angle_ind in these_indices]

    def choose_best_cut(self):
        """Identify minimum width metric, save it, also measure FWHM and dE/dx.
        """
        width = [cut.width_metric for cut in self.cuts]
        self.best_ind = np.argmin(width)
        self.best_cut = self.cuts[self.best_ind]
        self.fwhm_um = self.best_cut.measure_fwhm(self.options)
        self.dedx_kevum = self.best_cut.measure_dedx(self.options)

    def adjust_to_centroid(self):
        """Save position of best cut's centroid, and the energy there.
        """
        # final position and energy
        self.coordinates_pix = self.best_cut.find_centroid()
        self.energy_kev = self.info.interp(self.coordinates_pix[0],
                                           self.coordinates_pix[1])

    def measure_step_alpha(self):
        """Measure the direction from previous ridge point to this one.
        """
        if self.previous is None:
            # first point.
            self.step_alpha_deg = ((
                self.best_cut.angle_ind * self.options.angle_increment_deg) +
                180)
            if self.step_alpha_deg > 180:
                self.step_alpha_deg -= 360
        elif False:  # thought this was the best way to do it? but not MATLAB's
            # all subsequent points
            dpos = self.coordinates_pix - self.previous.coordinates_pix
            self.step_alpha_deg = 180/np.pi * np.arctan2(-dpos[1], -dpos[0])
            # the minus signs rotate 180 degrees.
            # Thus, step_alpha_deg is pointing backward, i.e. along the
            #   electron's motion, as opposed to toward the start of the track.
            # This does not require handling anywhere else, because
            #   estimate_next_step uses best_ind, not step_alpha_deg, for
            #   direction to the next step.
        else:
            # to match MATLAB calculation:
            # ignore centroid adjustment, i.e. just use the best cut direction.
            self.step_alpha_deg = (
                self.previous.best_cut.angle_ind *
                self.options.angle_increment_deg) + 180
        if self.step_alpha_deg > 360:
            self.step_alpha_deg -= 360

    def estimate_next_step(self):
        """Estimate the direction and position for the next step, in indices.
        """
        # from MATLAB code: next step is based on best angle index, not the
        #   position difference
        next_direction_ind = self.best_cut.angle_ind
        next_direction_rad = (
            next_direction_ind * np.pi/180 * self.options.angle_increment_deg)
        next_coordinates_pix = (
            self.coordinates_pix +
            self.options.position_step_size_pix * np.array(
                [np.cos(next_direction_rad), np.sin(next_direction_rad)]))

        return next_coordinates_pix, next_direction_ind


class Cut(object):
    """One cut for ridge following
    """

    def __init__(self, interp, options, coordinates_pix, angle_ind):
        """Define properties of the cut here.
        """

        self.angle_ind = angle_ind
        self.center = coordinates_pix
        self.set_coordinates(options.cut_coordinates[self.angle_ind])

        # interp2d object is supposed to take x and y vectors
        #   and make a 2D grid. which is not what I wanted... but it gives
        #   an input error if x and y are vectors.
        self.energy_kev = np.array(
            [float(interp(x, y)) for x, y in self.coordinates_pix])

        self.exclude_points(options)
        self.measure_width_metric(options)
        # self.measure_fwhm(self)
        # self.measure_dedx(self, options)
        # self.find_centroid(self)
        #

    def set_coordinates(self, cut_coordinates):
        """
        """
        x_cut, y_cut = zip(*cut_coordinates)
        x0, y0 = self.center
        self.coordinates_pix = zip(x0 + np.array(x_cut), y0 + np.array(y_cut))

    def exclude_points2(self, options):
        """
        DON'T exclude points, just add attributes for compatibility.
        """
        self.first_index_to_keep = 0
        self.first_index_to_lose = len(self.energy_kev)

    def exclude_points(self, options):
        """
        """
        thresh = options.cut_low_threshold_kev
        halves = [
            f(options.cut_distance_coordinate_pix, 0)
            for f in [np.less, np.greater]]
        energy_halves = [self.energy_kev[half] for half in halves]

        below_threshold = [energy < thresh for energy in energy_halves]
        if np.any(below_threshold[0]):
            # first [0]: first half. second [0]: first (only) dimension.
            list_of_indices_below_threshold = np.nonzero(below_threshold[0])[0]
            # first index to keep comes after last index below threshold
            first_index_to_keep = list_of_indices_below_threshold[-1] + 1
        else:
            first_index_to_keep = 0
        if np.any(below_threshold[1]):
            # [1]: second half. [0]: first (only) dimension.
            list_of_indices_below_threshold = np.nonzero(below_threshold[1])[0]
            first_index_to_lose = list_of_indices_below_threshold[0]
        else:
            first_index_to_lose = len(below_threshold[1])
        # in either case, first_index_to_lose is indexed for the half, instead
        #   of the whole cut.
        first_index_to_lose += len(energy_halves[0]) + 1

        self.coordinates_pix = self.coordinates_pix[
            first_index_to_keep:first_index_to_lose]
        self.energy_kev = self.energy_kev[
            first_index_to_keep:first_index_to_lose]

        self.first_index_to_keep = first_index_to_keep
        self.first_index_to_lose = first_index_to_lose

    def measure_width_metric(self, options):
        """
        """
        distance_cropped = options.cut_distance_from_center_pix[
            self.first_index_to_keep:self.first_index_to_lose]
        self.width_metric = np.sum(distance_cropped * self.energy_kev)

    def measure_fwhm(self, options):
        """Measure the FWHM of the cut (only the best cut of each step)
        """
        # MATLAB code uses HtFitCopy, not a real fit
        # just find the half-max on either side

        half_max = np.max(self.energy_kev) / 2
        max_index = np.argmax(self.energy_kev)

        # for each side, find the crossing point and interpolate linearly
        left_side_under = np.nonzero(self.energy_kev[:max_index] < half_max)[0]
        if len(left_side_under) > 0:
            left = left_side_under[-1]
            left_half_max = (
                left +
                (half_max - self.energy_kev[left]) /
                (self.energy_kev[left+1] - self.energy_kev[left]))
        else:
            # nothing under threshold - this is unusual
            left_half_max = 0

        right_side_under = np.nonzero(
            self.energy_kev[max_index+1:] < half_max)[0]
        if len(right_side_under) > 0:
            right = right_side_under[0] + max_index + 1
            right_half_max = (
                right -
                (half_max - self.energy_kev[right]) /
                (self.energy_kev[right-1] - self.energy_kev[right]))
        else:
            # nothing under threshold - this is unusual
            right_half_max = len(self.energy_kev)

        self.fwhm_um = (
            (right_half_max - left_half_max) *
            options.cut_sampling_interval_pix * options.pixel_size_um)

        return self.fwhm_um

    def measure_dedx(self, options):
        """Measure the dE/dx implied by this cut (only the best cut of the step)
        """
        self.dedx_kevum = (
            np.sum(self.energy_kev) * options.cut_sampling_interval_pix /
            options.pixel_size_um)

        return self.dedx_kevum

    def find_centroid(self):
        """
        """
        self.centroid_pix = np.average(
            np.transpose(self.coordinates_pix),
            weights=self.energy_kev,
            axis=1)

        return self.centroid_pix


def measure_track_dedx(ridge, options, start, end):
    """
    """

    dedx_values = [r.dedx_kevum for r in ridge[start:end]]
    measured_dedx_kevum = options.measurement_func(dedx_values)

    return measured_dedx_kevum


def measure_track_alpha(ridge, options, start, end):
    """
    """

    alpha_values = [r.step_alpha_deg for r in ridge[start:end]]
    # take care about the end of the circle...
    if np.any(alpha_values > 270) and np.any(alpha_values < 90):
        alpha_values[alpha_values < 90] += 360

    alpha_deg = options.measurement_func(alpha_values)
    if alpha_deg > 360:
        alpha_deg -= 360

    return alpha_deg


def compute_direction(energy_kev, options, info):
    """
    """

    # This part of the algorithm is a mess and should be re-written.
    # But until then, let's not change what has already been extensively
    #   evaluated and benchmarked...

    # 0. reverse indices (performed in ridge_follow)
    # 1. Measure width of track so we know how many points to skip
    # 2. Get measurement selection range, assuming beta==0 and no diffusion
    # 3. Get measurement selection range, for a given beta, and no diffusion
    # 4. First estimate, using beta=45
    # 5. Next selection calculation
    # 6. Second (final) estimate of beta, using beta==beta1
    # 7. Measure alpha
    # 8. construct output?

    dedx_ref = dedxref.dedx(energy_kev)

    # first estimate of beta, using initial guess of beta = 45
    start, end = select_measurement_points(
        info.ridge, options, energy_kev,
        beta_deg=options.initial_beta_guess_deg)
    dedx_meas = measure_track_dedx(info.ridge, options, start, end)
    first_cosbeta_estimate = np.minimum(dedx_ref / dedx_meas, 1)

    # second and final estimate of beta, using first estimate of beta
    start, end = select_measurement_points(
        info.ridge, options, energy_kev,
        cos_beta=first_cosbeta_estimate)
    dedx_meas = measure_track_dedx(info.ridge, options, start, end)
    cos_beta = np.minimum(dedx_ref / dedx_meas, 1)
    beta_deg = np.arccos(cos_beta) * 180/np.pi

    # measure alpha
    alpha_deg = measure_track_alpha(info.ridge, options, start, end)

    # outputs
    info.track_energy_kev = energy_kev
    info.dedx_meas_kevum = dedx_meas
    info.dedx_ref_kevum = dedx_ref
    info.measurement_start_pt = start
    info.measurement_end_pt = end
    info.alpha_deg = alpha_deg
    info.beta_deg = beta_deg


def diffusion_skip_points(ridge, options):
    """Measure track width over a few points, and calculate points to skip due
    to diffusion at the beginning of the track.
    """

    # points over which to measure width
    width_meas_length_pts = np.round(
        options.width_measurement_len_pix / options.position_step_size_pix)
    width_meas_length_pts = np.maximum(width_meas_length_pts, 1)
    width_meas_length_pts = np.minimum(width_meas_length_pts, len(ridge))
    width_meas_length_pts = int(width_meas_length_pts)  # used as index
    width_values = [r.fwhm_um for r in ridge[:width_meas_length_pts]]

    # measure width
    measured_width_um = options.measurement_func(width_values)

    # number of points to skip from diffusion (from logbook, 10/29/2009)
    n_points_to_skip_from_diffusion = (
        (measured_width_um - options.pixel_size_um) /
        (options.pixel_size_um * options.position_step_size_pix))
    n_points_to_skip_from_diffusion = np.maximum(
        0, n_points_to_skip_from_diffusion)

    return n_points_to_skip_from_diffusion


def base_measurement_points(energy_kev):
    """Base measurement point selection assumes beta=0 and no diffusion.
    """

    # equivalent of HtSelectMeasurementPoints in MATLAB code.
    # reference: node 1875 on the old bearing.berkeley.edu website
    start = np.sqrt(0.0825 * energy_kev + 15.814) - 3.4
    start -= 1      # change in which point each step is associated with
    start = np.maximum(start, 0)
    end = start * 2 + 3.4
    end -= 1        # change in which point each step is associated with
    end = np.maximum(end, 0)

    start -= 1      # python is 0-based, MATLAB is 1-based
    # end doesn't change:
    #   python is 0-based, MATLAB is 1-based, BUT python excludes endpoint
    return start, end


def select_measurement_points(ridge, options, energy_kev, beta_deg=None,
                              cos_beta=None):
    """
    """

    if beta_deg is None and cos_beta is None:
        raise ValueError('select_measurement_points needs more information')
    if cos_beta is None:
        cos_beta = np.cos(np.pi/180 * beta_deg)

    # pdb.set_trace()
    n_points_to_skip_from_diffusion = diffusion_skip_points(ridge, options)
    base_start, base_end = base_measurement_points(energy_kev)

    start = np.ceil(
        (base_start + 1) * cos_beta + n_points_to_skip_from_diffusion) - 1
    end = np.ceil(base_end * cos_beta + n_points_to_skip_from_diffusion)
    start = np.minimum(start, len(ridge) - 1)
    end = np.minimum(end, len(ridge))
    start = np.maximum(start, 0)
    end = np.maximum(end, start + 1)

    return int(start), int(end)     # these become indices


def measurement_debug(options, info, verbosity=1, MATLAB=False):
    """
    """
    energy = np.sum(info.prepared_image_kev)
    beta0 = options.initial_beta_guess_deg
    dedx_ref = info.dedx_ref_kevum
    print(' ')
    if not MATLAB:
        print('Ridge length: {}'.format(len(info.ridge)))
        if verbosity > 0:
            print('Diffusion skip points: {:.4f}'.format(
                diffusion_skip_points(info.ridge, options)))
            print('Base measurement points: ({:.4f}, {:.4f})'.format(
                *base_measurement_points(energy)))
        start, end = select_measurement_points(
            info.ridge, options, energy, beta_deg=beta0)
        print('First selection: ({:d}, {:d})'.format(start, end))
        dedx1 = measure_track_dedx(info.ridge, options, start, end)
        cb = np.minimum(dedx_ref / dedx1, 1)
        if verbosity > 0:
            print('dE/dx estimate: {:.4f}'.format(dedx1))
            print('Cos(beta) estimate: {:.4f}'.format(cb))
        start, end = select_measurement_points(
            info.ridge, options, energy, cos_beta=cb)
        print('Second selection: ({:d}, {:d})'.format(start, end))
        if verbosity > 0:
            print('Alpha values:')
            for i, r in enumerate(info.ridge[start:end]):
                print('  Step {}: {:d}'.format(i+start, r.step_alpha_deg))
        print('Alpha measurement: {:.1f}'.format(info.alpha_deg))
        print('Beta measurement: {:.4f}'.format(info.beta_deg))
    else:
        # MATLAB
        pass


def set_output(image, options, info):
    """
    """
    pass


# # # # # # # # # # # # # # # # # # # #
#           Unit test stuff           #
# # # # # # # # # # # # # # # # # # # #
def test_input():
    """Create sample electron track image for testing.
    """

    # this is a track:
    # Mat_CCD_SPEC_500k_49_TRK_truncated.mat, Event 8

    image = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            -0.0051609, 0.053853, 0.095667, -0.0033475],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0.021966, 0.69575, 0.75888, 0.048261],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            -0.033524, 0.28492, 1.9392, 1.0772, 0.056027],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            -0.018537, 1.0306, 3.7651, 1.0724, 0.020338],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0.0072011, 0.98876, 3.4822, 0.89705, 0.018026],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0.022901, 0.45766, 1.9979, 0.71425, 0.046171],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0.19834, 1.4178, 1.1684, 0.10283],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.065909, 0.056041, 0.080346, 0.033988,
            0.011497, 0.026686, 0.058919, 0.053537, 0.10681, 0.13619, 0.14899,
            0.062984, 0.0047364, -0.0017255, 0.10294, 1.3059, 1.5101, 0.14706],
        [0, 0, 0, 0, 0.0053918, 0.0035012, 0.021259, 0.066659, 0.19352,
            0.48743, 0.91541, 1.0855, 0.92411, 0.52781, 0.44985, 0.73201,
            1.1867, 1.406, 1.7395, 2.4826, 1.3234, 0.62977, 0.29166, 0.26536,
            2.1381, 2.0943, 0.16611],
        [0, 0, 0.010123, 0.11812, 0.41158, 0.63741, 0.82542, 1.1754, 1.9752,
            2.9183, 3.7162, 4.472, 4.9214, 3.8307, 2.4756, 2.0333, 1.6727,
            1.4002, 1.8243, 2.5997, 1.9335, 1.8506, 1.6702, 1.7379, 3.2162,
            1.9279, 0.080845],
        [0, 0, 0.0094986, 0.84334, 3.1883, 4.4307, 5.1898, 5.8549, 5.6754,
            4.5784, 4.2334, 5.1794, 6.554, 5.0217, 2.0617, 0.67738, 0.20441,
            0.13334, 0.11854, 0.26303, 0.30459, 0.55708, 1.1594, 2.2099, 2.844,
            0.91984, -0.0042292],
        [0, 0, 0.063279, 1.1166, 4.1508, 5.491, 5.7499, 5.6155, 3.743, 1.6617,
            1.0403, 1.47, 2.8975, 2.7051, 0.93501, 0.1143, 0.0035993, 0, 0, 0,
            0.0051778, 0.032716, 0.06831, 0.27357, 0.38092, 0.072979,
            0.035218],
        [0, 0, 0.0081916, 0.23584, 0.93687, 1.3889, 1.3397, 1.0887, 0.57792,
            0.32439, 0.55854, 1.261, 2.1775, 1.9507, 0.65327, 0.10457, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0.024259, 0.022813, 0.041052, 0.077169, 0.13036, 0.38906,
            1.1307, 2.0968, 2.6917, 2.5884, 1.5493, 0.35102, 0.026962, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0.013108, 0.037261, 0.060011, 0.035246, 0.073982, 0.2227, 0.60201,
            1.4271, 2.7482, 3.4362, 3.0109, 1.6935, 0.6734, 0.12314, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0.018483, 0.19417, 0.86795, 1.1453, 0.75306, 0.55931, 1.0736, 1.9404,
            2.6781, 3.1387, 2.4657, 1.3213, 0.47934, 0.12477, 0.002662, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0.13897, 1.829, 6.6556, 8.0879, 4.4295, 2.3569, 2.599, 2.9012, 2.559,
            1.6695, 0.80228, 0.23316, 0.072496, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0],
        [0.42121, 4.9934, 16.3581, 17.7758, 8.5629, 3.489, 2.5868, 2.0779,
            1.1432, 0.43788, 0.095359, 0.029513, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0],
        [0.34745, 3.7962, 12.1618, 12.3907, 5.4372, 1.9508, 1.0099, 0.59748,
            0.22756, 0.081068, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0],
        [0.093006, 0.81449, 2.6296, 2.5491, 1.0871, 0.35421, 0.15411, 0.0413,
            0.015189, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0.020778, 0.038066, 0.17076, 0.1381, 0.056419, -0.0060239, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    return np.array(image)


if __name__ == '__main__':
    """
    Run tests.
    """

    image = test_input()

    __, info = reconstruct(image)

    print('')
    if True:
        print('Track energy: {:.2f}'.format(info.track_energy_kev))
        print('')
    if False:
        print('Low threshold used: {}'.format(info.threshold_used))
    if False:
        print('Binary image:')
        print(info.binary_image + 0)
        print('')
    if False:
        print('Thinned image:')
        print(info.thinned_image + 0)
        print('')
    if False:
        print('ends_xy:')
        print(info.ends_xy)
        print('')
    if False:
        print('start_coordinates:')
        print(info.start_coordinates)
        print('')
    if False:
        print('start_direction_deg:')
        print(info.start_direction_deg)
        print('')
    if False:
        print('ridge point coordinates:')
        for r in info.ridge:
            print('({:.4f}, {:.4f})').format(*r.coordinates_pix)
        print('')
    if True:
        print('steps alpha:')
        for r in info.ridge:
            if r.step_alpha_deg is not None:
                print('{:.4f}'.format(r.step_alpha_deg))
            else:
                print('None')
        print('')
    if True:
        print('dedx_meas, dedx_ref:')
        print('{:.4f}, {:.4f}'.format(
            info.dedx_meas_kevum, info.dedx_ref_kevum))
        print('')
    if True:
        print('measurement start and end:')
        print(info.measurement_start_pt, info.measurement_end_pt)
        print('')
    if True:
        print('measured alpha, beta:')
        print('{:.2f}, {:.2f}'.format(info.alpha_deg, info.beta_deg))
        print('')

    if False:
        trackplot.plot_track_image(info.prepared_image_kev)
