#!/bin/python

import numpy as np
import ipdb as pdb
import scipy.ndimage

import dedxref
import thinning



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
    track_energy_kev, prepared_image_kev = prepare_image(
        original_image_kev, options)

    # low threshold, thinning, identify ends
    info = ReconstructionInfo()
    choose_initial_end(prepared_image_kev, options, info)

    # ridge following
    ridge_follow(prepared_image_kev, options, info)

    compute_direction(track_energy_kev, options, info)

    output = None   # temp
    return output, info
    # return set_output(prepared_image_kev, edge_segments, ridge, measurement,
    #                   options)


class ReconstructionOptions():
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
        cut0_x = np.zeros_like(cut0_y)
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

class ReconstructionInfo():
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

class ReconstructionOutput():
    """Electron track result from HybridTrack algorithm.
    """

class TrackException(Exception): pass

class NoEndsFound(TrackException): pass

class InfiniteLoop(TrackException): pass


def construct_energy_kernel(radius_um, pixel_size_um):
    """Construct kernel for energy measurement.
    """

    radius_pix = radius_um / pixel_size_um
    ceil_radius_pix = int(np.ceil(radius_pix) - 1)
    coordinates = range(-ceil_radius_pix, ceil_radius_pix+1)
    kernel = [[0+(x**2+y**2 < radius_pix**2) for x in coordinates]
                                             for y in coordinates]
    return np.array(kernel)


def prepare_image(image_kev, options):
    """Add a buffer of zeros around the original image.
    """

    # imageEdgeBuffer does not need to handle the cutTotalLength/2 in any direction.
    #   it only needs to handle the ridge points going off the edge.

    # TODO: reduce buffer_width to... one pixel?

    buffer_width_um = 0.55 * options.cut_total_length_pix
    buffer_width_pix = np.ceil(buffer_width_um / options.pixel_size_um)
    orig_size = np.array(np.shape(image_kev))
    new_image_size = (orig_size + 2*buffer_width_pix*np.ones(2))
    new_image_kev = np.zeros(new_image_size)

    indices_of_original = [
                           [int(it) + int(buffer_width_pix)
                            for it in range(orig_size[dim])]
                           for dim in [0,1]]
    # pdb.set_trace()
    new_image_kev[np.ix_(indices_of_original[0], indices_of_original[1])
                  ] = image_kev

    track_energy_kev = np.sum(np.sum(image_kev))

    return track_energy_kev, new_image_kev


def choose_initial_end(image_kev, options, info):
    """
    locate_all_ends
    measure_energies
    get_starting_point
    """

    def locate_all_ends(image_kev, options, info):
        """Apply threshold, perform thinning, identify ends.
        """
        ends_image = [False]
        current_threshold = options.low_threshold_kev
        connectivity = np.ones((3,3))

        # normally this while loop only runs once.
        # if the track is a loop and so no ends are found, then increase the
        #   threshold until an end is found.
        # TODO: does this actually work?

        while np.all(np.logical_not(ends_image)) and (
                current_threshold <= 10*options.low_threshold_kev):
            binary_image = image_kev > current_threshold
            thinned_image = thinning.thin(binary_image)+0
            num_neighbors_image = scipy.ndimage.convolve(
                thinned_image, np.ones((3,3)),mode='constant',cval=0.0) - 1
            num_neighbors_image = num_neighbors_image * thinned_image
            ends_image = num_neighbors_image==1

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
            # Rather than raising an exception here, allow this to return.
            info.threshold_used = np.nan
            info.ends_xy = [[],[]]
            info.error = 'no ends found'


    def measure_energies(image_kev, options, info):
        """Compute energies using a kernel, for each end.
        """

        kernel = options.energy_kernel
        energy_convolved_image = scipy.ndimage.convolve(
            image_kev, kernel, mode='constant',cval=0.0)
        ends_energy = []
        for (x,y) in info.ends_xy:
            ends_energy.append(energy_convolved_image[x,y])

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
            temp_image[xy[0],xy[1]] = 0
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
        info.start_direction_deg = (180/np.pi *
            np.arctan2(-step_xy[1], -step_xy[0]))
        if info.start_direction_deg < 0:
            info.start_direction_deg += 360
        info.start_direction_ind = (info.start_direction_deg /
                                      options.angle_increment_deg)

    # main
    locate_all_ends(image_kev, options, info)
    if info.error == 'no ends found':
        raise NoEndsFound

    measure_energies(image_kev, options, info)
    get_starting_point(options, info)



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

# # # # # # # # # # # # # # # # # # # #
#           Unit test stuff           #
# # # # # # # # # # # # # # # # # # # #
def test_input():
    """Create sample electron track image for testing.
    """

    # this is a track:
    # Mat_CCD_SPEC_500k_49_TRK_truncated.mat, Event 8

    image = [
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-0.0051609,0.053853,0.095667,-0.0033475],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.021966,0.69575,0.75888,0.048261],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-0.033524,0.28492,1.9392,1.0772,0.056027],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-0.018537,1.0306,3.7651,1.0724,0.020338],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.0072011,0.98876,3.4822,0.89705,0.018026],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.022901,0.45766,1.9979,0.71425,0.046171],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.19834,1.4178,1.1684,0.10283],
        [0,0,0,0,0,0,0,0,0,0.065909,0.056041,0.080346,0.033988,0.011497,0.026686,0.058919,0.053537,0.10681,0.13619,0.14899,0.062984,0.0047364,-0.0017255,0.10294,1.3059,1.5101,0.14706],
        [0,0,0,0,0.0053918,0.0035012,0.021259,0.066659,0.19352,0.48743,0.91541,1.0855,0.92411,0.52781,0.44985,0.73201,1.1867,1.406,1.7395,2.4826,1.3234,0.62977,0.29166,0.26536,2.1381,2.0943,0.16611],
        [0,0,0.010123,0.11812,0.41158,0.63741,0.82542,1.1754,1.9752,2.9183,3.7162,4.472,4.9214,3.8307,2.4756,2.0333,1.6727,1.4002,1.8243,2.5997,1.9335,1.8506,1.6702,1.7379,3.2162,1.9279,0.080845],
        [0,0,0.0094986,0.84334,3.1883,4.4307,5.1898,5.8549,5.6754,4.5784,4.2334,5.1794,6.554,5.0217,2.0617,0.67738,0.20441,0.13334,0.11854,0.26303,0.30459,0.55708,1.1594,2.2099,2.844,0.91984,-0.0042292],
        [0,0,0.063279,1.1166,4.1508,5.491,5.7499,5.6155,3.743,1.6617,1.0403,1.47,2.8975,2.7051,0.93501,0.1143,0.0035993,0,0,0,0.0051778,0.032716,0.06831,0.27357,0.38092,0.072979,0.035218],
        [0,0,0.0081916,0.23584,0.93687,1.3889,1.3397,1.0887,0.57792,0.32439,0.55854,1.261,2.1775,1.9507,0.65327,0.10457,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0.024259,0.022813,0.041052,0.077169,0.13036,0.38906,1.1307,2.0968,2.6917,2.5884,1.5493,0.35102,0.026962,0,0,0,0,0,0,0,0,0,0,0],
        [0,0.013108,0.037261,0.060011,0.035246,0.073982,0.2227,0.60201,1.4271,2.7482,3.4362,3.0109,1.6935,0.6734,0.12314,0,0,0,0,0,0,0,0,0,0,0,0],
        [0.018483,0.19417,0.86795,1.1453,0.75306,0.55931,1.0736,1.9404,2.6781,3.1387,2.4657,1.3213,0.47934,0.12477,0.002662,0,0,0,0,0,0,0,0,0,0,0,0],
        [0.13897,1.829,6.6556,8.0879,4.4295,2.3569,2.599,2.9012,2.559,1.6695,0.80228,0.23316,0.072496,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0.42121,4.9934,16.3581,17.7758,8.5629,3.489,2.5868,2.0779,1.1432,0.43788,0.095359,0.029513,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0.34745,3.7962,12.1618,12.3907,5.4372,1.9508,1.0099,0.59748,0.22756,0.081068,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0.093006,0.81449,2.6296,2.5491,1.0871,0.35421,0.15411,0.0413,0.015189,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0.020778,0.038066,0.17076,0.1381,0.056419,-0.0060239,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
    return np.array(image)


if __name__ == '__main__':
    """
    Run tests.
    """

    image = test_input()

    __, info = reconstruct(image)

    print('')
    if True:
        print('Low threshold used: {}'.format(info.threshold_used))
    if False:
        print('Binary image:')
        print(info.binary_image+0)
        print('')
    if False:
        print('Thinned image:')
        print(info.thinned_image+0)
        print('')
    if True:
        print('ends_xy:')
        print(info.ends_xy)
        print('')
    if True:
        print('start_coordinates:')
        print(info.start_coordinates)
        print('')
    if True:
        print('start_direction_deg:')
        print(info.start_direction_deg)
        print('')
    
