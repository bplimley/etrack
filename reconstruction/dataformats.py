#!/usr/bin/python

import numpy as np


def get_format(class_name):
    """
    Retrieve the data format stored for class_name.

    Input:
      class_name: string representing the class. self.class_name
    """

    if class_name == 'AlgorithmResults':
        data_format = (
            ClassAttr('parent', 'AlgorithmResults',
                      may_be_none=True, is_user_object=True,
                      is_always_list=True),
            ClassAttr('filename', str,
                      may_be_none=True, is_always_list=True),
            ClassAttr('has_alpha', bool),
            ClassAttr('has_beta', bool),
            ClassAttr('data_length', int),
            ClassAttr('uncertainty_list', 'Uncertainty',
                      is_user_object=True, is_always_list=True),
            ClassAttr('alpha_true_deg', np.ndarray,
                      may_be_none=True, make_dset=True),
            ClassAttr('alpha_meas_deg', np.ndarray,
                      may_be_none=True, make_dset=True),
            ClassAttr('beta_true_deg', np.ndarray,
                      may_be_none=True, make_dset=True),
            ClassAttr('beta_meas_deg', np.ndarray,
                      may_be_none=True, make_dset=True),
            ClassAttr('energy_tot_kev', np.ndarray,
                      may_be_none=True, make_dset=True),
            ClassAttr('energy_dep_kev', np.ndarray,
                      may_be_none=True, make_dset=True),
            ClassAttr('depth_um', np.ndarray,
                      may_be_none=True, make_dset=True),
            ClassAttr('is_contained', np.ndarray,
                      may_be_none=True, make_dset=True),
        )

    elif class_name == 'Uncertainty':
        data_format = (
            ClassAttr('delta', np.ndarray, make_dset=True),
            ClassAttr('n_values', int),
            ClassAttr('metrics', 'UncertaintyParameter',
                      is_always_dict=True, is_user_object=True),
            ClassAttr('angle_type', str),
        )

    elif class_name == 'AlphaGaussPlusConstant':
        data_format = list(get_format('Uncertainty'))
        data_format.extend([
            ClassAttr('nhist', np.ndarray, make_dset=True),
            ClassAttr('xhist', np.ndarray, make_dset=True),
            ClassAttr('resolution', float),
        ])
        # TODO: fit?
        data_format = tuple(data_format)

    elif class_name == 'AlphaGaussPlusConstantPlusBackscatter':
        data_format = get_format('AlphaGaussPlusConstant')

    elif class_name == 'UncertaintyParameter':
        data_format = (
            ClassAttr('name', str),
            ClassAttr('fit_name', str),
            ClassAttr('value', float),
            ClassAttr('uncertainty', float, is_sometimes_list=True),
            ClassAttr('units', str),
            ClassAttr('axis_min', float),
            ClassAttr('axis_max', float),
        )

    elif class_name == 'G4Track':
        raise Exception('G4Track data format not defined yet')

    elif class_name == 'Track':
        data_format = (
            ClassAttr('is_modeled', bool),
            ClassAttr('pixel_size_um', float),
            ClassAttr('noise_ev', float, may_be_none=True),
            ClassAttr('g4track', 'G4Track',
                      may_be_none=True, is_user_object=True),
            ClassAttr('energy_kev', float),
            ClassAttr('x_offset_pix', int, may_be_none=True),
            ClassAttr('y_offset_pix', int, may_be_none=True),
            ClassAttr('timestamp', str, may_be_none=True),
            ClassAttr('shutter_ind', int, may_be_none=True),
            ClassAttr('label', str, may_be_none=True),
        )

    else:
        raise Exception('Unknown class_name')

    return data_format


class ClassAttr(object):
    """
    Description of one attribute of a class, for the purposes of saving to file
    and loading from file.
    """

    __version__ = '0.1'

    def __init__(self, name, dtype,
                 make_dset=False,
                 may_be_none=False,
                 is_always_list=False,
                 is_sometimes_list=False,
                 is_always_dict=False,
                 is_sometimes_dict=False,
                 is_user_object=False):
        self.name = name
        self.dtype = dtype
        self.make_dset = make_dset
        self.may_be_none = may_be_none
        self.is_always_list = is_always_list
        self.is_sometimes_list = is_sometimes_list
        self.is_always_dict = is_always_dict
        self.is_sometimes_dict = is_sometimes_dict
        self.is_user_object = is_user_object