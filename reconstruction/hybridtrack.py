#!/bin/python

import numpy as np

import dedxref


def reconstruct(original_image_kev,
        pixel_size_um = 10.5):
    """
    Perform trajectory reconstruction on CCD electron track image.

    HybridTrack algorithm, copied from MATLAB code, 2015-08-26.

    Inputs:
      pixel_size_um: pitch of pixels (default 10.5 um)

    Output:
      TBD.
    """

    # currently, only default options are supported, set in class def
    options = ReconstructionOptions(pixel_size_um)

    # add buffer of zeros around image
    track_energy_kev, prepared_image_kev, options = prepare_image(
        original_image_kev, options)

    # low threshold, thinning, identify ends
    edge_segments = choose_initial_end(prepared_image_kev, options)

    if edge_segments.chosen_index is None:
        # no end found
        # exit unsuccessfully
        pass

    # ridge following
    try:
        ridge = ridge_follow(prepared_image_kev, edge_segments, options)
    except:
        # infinite loop error
        pass

    measurement, ridge = compute_direction(track_energy_kev, ridge, options)

    return set_output(prepared_image_kev, edge_segments, ridge, measurement,
                      options)


class ReconstructionOptions():
    """
    Reconstruction options for HybridTrack algorithm.

    Everything is set to defaults (based on pixel size) upon initialization.
    """

    def __init__(self, pixel_size_um):
        self.pixel_size_um = pixel_size_um
        self.set_low_threshold()
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
        self.angle_increment_rad = self.angle_increment_deg * np.pi/180
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
        self.search_angle_indices = (search_angle_deg /
                                     self.angle_increment_deg)

        pi_deg = 180
        self.pi_indices = pi_deg / self.angle_increment_deg

        # all possible cut angles, for pre-calculating cuts
        #   *different from MATLAB code: starts with 0 instead of angle_incr
        self.cut_angle_rad = np.arange(0, 2*np.pi, self.angle_increment_rad)
        self.cut_angle_deg = np.arange(0, 2*pi_deg, self.angle_increment_deg)

        # other methods include ... what in python?
        self.cut_interpolation_method = 'linear'

        # Now, we define the x and y coordinates of the samples along a cut,
        #   and rotate to all possible angle increments.
        # The 0-degree cut is along the y-axis because the electron ridge is
        #   along the x-axis.
        cut0_y = np.arange(-self.cut_total_length_pix/2,
                          self.cut_total_length_pix/2+1e-3, # includes endpoint
                          self.cut_sampling_interval_pix)
        cut0_x = np.zeros_like(cut0y)
        cut0_xy = np.array([cut0_x,cut0_y]).T
        # rotation matrices
        R = [np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]])
             for th in self.cut_angle_rad]
        self.cut_xy = [cut0_xy.dot(Ri) for Ri in R]     # matrix multiplication
        # cut_distance_from_center is used for the width metric
        self.cut_distance_from_center_pix = np.abs(cut0_y)
        # cut_distance_coordinate is used for ???
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
        self.infinite_loop_threshold_pix = self.position_step_size / 2

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


def prepare_image(original_image_kev, options):
    """
    """
    pass


def choose_initial_end(prepared_image_kev, options):
    """
    """
    pass


class EdgeSegments():
    """
    """

    def __init__(self):
        # ?
        pass


def ridge_follow(prepared_image_kev, edge_segments, options):
    """
    """
    pass


class Ridge():
    """
    """

    def __init__(self):
        # ?
        pass


def compute_direction(track_energy_kev, ridge, options):
    """
    """
    pass


def set_output(prepared_image_kev, edge_segments, ridge, measureemnt, options):
    """
    """
    pass
