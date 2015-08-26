#!/bin/python

import numpy as np

import dedxref


def reconstruct(original_image_kev,
        pixel_size_um = 10.5,
        low_threshold_kev = None):
    """
    Perform trajectory reconstruction on CCD electron track image.

    HybridTrack algorithm, copied from MATLAB code, 2015-08-26.

    Inputs:
      pixel_size_um: pitch of pixels (default 10.5 um)
      low_threshold_kev: threshold to apply for binary image.
        (default is 0.5 keV for 10.5 um, scaled by pixel area for other sizes)

    Output:
      TBD.
    """

    # set default low threshold
    if low_threshold_kev is None:
        # assign default based on pixel size
        base_low_threshold_kev = 0.5
        scaling_factor = pixel_size_um**2 / 10.5**2
        low_threshold_kev = scaling_factor * base_low_threshold_kev

    options = ReconstructionOptions()
    track_energy_kev, prepared_image_kev, options = prepare_image(
        original_image_kev, options)
    edge_segments = choose_initial_end(prepared_image_kev, options)

    if edge_segments.chosen_index is None:
        # no end found
        # exit unsuccessfully
        pass

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
    """

    def __init__(self):
        # assign all properties here
        pass


def prepare_image(original_image_kev, options):
    """
    """
    pass


def choose_initial_end(prepared_image_kev, options):
    """
    """
    pass


def ridge_follow(prepared_image_kev, edge_segments, options):
    """
    """
    pass


def compute_direction(track_energy_kev, ridge, options):
    """
    """
    pass


def set_output(prepared_image_kev, edge_segments, ridge, measureemnt, options):
    """
    """
    pass
