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
            ClassAttr('alpha_unc', 'Uncertainty',
                      is_user_object=True, may_be_none=True),
            ClassAttr('beta_unc', 'Uncertainty',
                      is_user_object=True, may_be_none=True),
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

    elif class_name == 'Alpha68':
        data_format = get_format('Uncertainty')

    elif class_name == 'BetaRms':
        data_format = get_format('Uncertainty')

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
        data_format = (
            ClassAttr('matrix', np.ndarray, make_dset=True),
            ClassAttr('x', np.ndarray, make_dset=True),
            ClassAttr('dE', np.ndarray, make_dset=True),
            ClassAttr('x0', np.ndarray, may_be_none=True),
            ClassAttr('alpha_deg', float, may_be_none=True),
            ClassAttr('beta_deg', float, may_be_none=True),
            ClassAttr('first_step_vector', np.ndarray, may_be_none=True),
            ClassAttr('energy_tot_kev', float, may_be_none=True),
            ClassAttr('energy_dep_kev', float, may_be_none=True),
            ClassAttr('energy_esc_kev', float, may_be_none=True),
            ClassAttr('energy_xray_kev', float, may_be_none=True),
            ClassAttr('energy_brems_kev', float, may_be_none=True),
            ClassAttr('depth_um', float, may_be_none=True),
            ClassAttr('is_contained', bool, may_be_none=True),
        )

    elif class_name == 'Track':
        data_format = (
            ClassAttr('image', np.ndarray, make_dset=True),
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
            ClassAttr('algorithms', 'AlgorithmOutput',
                      is_user_object=True, is_always_dict=True),
        )

    elif class_name == 'AlgorithmOutput':
        data_format = (
            ClassAttr('alg_name', str),
            ClassAttr('alpha_deg', float),
            ClassAttr('beta_deg', float),
        )

    elif class_name == 'AlgorithmOutputMatlab':
        data_format = (
            ClassAttr('alg_name', str),
            ClassAttr('alpha_deg', float),
            ClassAttr('beta_deg', float),
            ClassAttr('info', 'MatlabAlgorithmInfo',
                      is_user_object=True),
        )

    elif class_name == 'MatlabAlgorithmInfo':
        data_format = (
            ClassAttr('Tind', int),
            ClassAttr('lt', float),
            ClassAttr('n_ends', int),
            ClassAttr('Eend', float),
            ClassAttr('alpha', float),
            ClassAttr('beta', float),
            ClassAttr('dalpha', float),
            ClassAttr('dbeta', float),
            ClassAttr('edgesegments_energies_kev', np.ndarray),
            ClassAttr('edgesegments_coordinates_pix', np.ndarray),
            ClassAttr('edgesegments_chosen_index', int),
            ClassAttr('edgesegments_start_coordinates_pix', np.ndarray),
            ClassAttr('edgesegments_start_direction_indices', float),
            ClassAttr('edgesegments_low_threshold_used', float),
            ClassAttr('dedx_ref', float),
            ClassAttr('dedx_meas', float),
            ClassAttr('measurement_start_ind', int),
            ClassAttr('measurement_end_ind', int),
        )

    elif class_name == 'MomentsReconstruction':
        data_format = (
            ClassAttr('original_image_kev', np.ndarray, make_dset=True),
            ClassAttr('pixel_size_um', float),
            ClassAttr('starting_distance_um', float),
            ClassAttr('ends_energy', np.ndarray),
            ClassAttr('rough_est', float),
            ClassAttr('error', str, may_be_none=True),
            ClassAttr('box_x', np.ndarray, may_be_none=True),
            ClassAttr('box_y', np.ndarray, may_be_none=True),
            ClassAttr('edge_pixel_count', int, may_be_none=True),
            ClassAttr('edge_pixel_segments', int, may_be_none=True),
            ClassAttr('phi', np.float, may_be_none=True),
            ClassAttr('R', np.float, may_be_none=True),
            ClassAttr('alpha', np.float, may_be_none=True),
            ClassAttr('x0', np.ndarray, may_be_none=True),
        )

    elif class_name == 'Classifier':
        data_format = (
            ClassAttr('g4track', 'G4Track', is_user_object=True),
            ClassAttr('scatterlen_um', float, may_be_none=True),
            ClassAttr('overlapdist_um', float, may_be_none=True),
            ClassAttr('scatter_type', str, may_be_none=True),
            ClassAttr('use2d_angle', bool, may_be_none=True),
            ClassAttr('use2d_dist', bool, may_be_none=True),
            ClassAttr('angle_threshold_deg', float, may_be_none=True),
            ClassAttr('escaped', bool, may_be_none=True),
            ClassAttr('wrong_end', bool, may_be_none=True),
            ClassAttr('early_scatter', bool, may_be_none=True),
            ClassAttr('total_scatter_angle', float, may_be_none=True),
            ClassAttr('overlap', bool, may_be_none=True),
            ClassAttr('n_ends', int, may_be_none=True),
            ClassAttr('max_end_energy', float, may_be_none=True),
            ClassAttr('min_end_energy', float, may_be_none=True),
            ClassAttr('error', str, may_be_none=True),
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
