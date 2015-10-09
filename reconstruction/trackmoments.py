#!/usr/bin/python

import numpy as np

import hybridtrack


def reconstruct(original_image_kev, pixel_size_um=10.5):
    """
    """

    options = hybridtrack.ReconstructionOptions(pixel_size_um)
    info = hybridtrack.ReconstructionInfo()

    hybridtrack.choose_initial_end(original_image_kev, options, info)

    segment_initial_end(original_image_kev, options, info)

    compute_moments(options, info)

    compute_direction(options, info)

    return options, info


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
