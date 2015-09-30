#!/usr/bin/python

import numpy as np
import lmfit


def delta_alpha(alpha1_deg, alpha2_deg):
    """
    Compute alpha2 - alpha1, returning value on (-180, +180].

    scalar, scalar: return a scalar
    vector, scalar: return a vector (each vector value compared to scalar)
    vector, vector: vectors should be same size. compare elementwise.
    """

    # type conversion... also copies the data to avoid modifying the inputs
    alpha1_deg = np.array(alpha1_deg)
    alpha2_deg = np.array(alpha2_deg)

    dalpha = alpha2_deg - alpha1_deg
    adjust_dalpha(dalpha)

    return dalpha


def adjust_dalpha(dalpha):
    """
    Put all values into (-180, +180].
    """

    if type(dalpha) is np.ndarray:
        # elementwise correction
        while np.any(dalpha > 180):
            dalpha[dalpha > 180] -= 360
        while np.any(dalpha <= -180):
            dalpha[dalpha <= -180] += 360
    else:
        # scalar
        while dalpha > 180:
            dalpha -= 360
        while dalpha <= -180:
            dalpha += 360



def delta_beta(beta_true_deg, beta_alg_deg):
    """
    Compute beta_alg_deg - abs(beta_true_deg).

    scalar, scalar: return a scalar
    vector, scalar: return a vector (each vector value compared to scalar)
    vector, vector: vectors should be same size. compare elementwise.
    """

    # type conversion
    beta_true_deg = np.array(beta_true_deg.copy())
    beta_alg_deg = np.array(beta_alg_deg.copy())

    dbeta = beta_alg_deg - np.abs(beta_true_deg)

    return dbeta


class DataWarning(UserWarning):
    pass


class AlgorithmUncertainty(object):
    """
    Produce and store the alpha and beta uncertainty metrics.

    Minimum input: dalpha OR (beta_true AND beta_meas)

    mode: sets both alpha_mode and beta_mode simultaneously.
      (default: mode=2)

    alpha_mode:
      1: 68% metrics (not coded yet)
      2: forward peak, constant (random) background
      3: forward peak, backscatter peak, constant background

    beta_mode:
      1: 68% containment
      2: RMS
      3: zero fraction, ???
    """

    def __init__(self, dalpha=None, beta_true=None, beta_meas=None,
                 mode=2, alpha_mode=None, beta_mode=None):
        if dalpha is None and beta_true is None and beta_meas is None:
            raise RuntimeError('AlgorithmUncertainty requires ' +
                'either dalpha, or both beta_true and beta_meas')
        elif np.logical_xor(beta_true is None, beta_meas is None):
            raise RuntimeError('Both beta_true and beta_meas must be provided')
        if alpha_mode is None:
            alpha_mode = mode
        if beta_mode is None:
            beta_mode = mode

        if dalpha is not None:
            alpha_result = fit_alpha(dalpha, mode=alpha_mode)

        if beta_true is not None:
            beta_result = fit_beta(beta_true=beta_true, beta_meas=beta_meas,
                mode=beta_mode)




def fit_alpha(dalpha, mode=2):
    """
    Return a metric for alpha uncertainty.

    Mode:
    1: 68% (one parameter)
    2: FWHM and peak efficiency (two parameter)
    3: 2 plus backscatter peak (three parameter)
    """

    dalpha = np.array(dalpha)
    if np.any(np.isnan(dalpha)):
        raise DataWarning('NaN values in dalpha')
        dalpha = dalpha(not np.isnan(dalpha))
    if np.any(np.isinf(dalpha)):
        raise DataWarning('Inf values in dalpha')
        dalpha = dalpha(not np.isinf(dalpha))
    adjust_dalpha(dalpha)
    dalpha = np.abs(dalpha.flatten)

    n_values = len(dalpha)
    resolution = np.minimum(100 * 180 / n_values, 15)   # from MATLAB
    n_bins = 180 / resolution
    nhist, edges = np.histogram(dalpha, bins=n_bins, range=(0.0,180.0))
    bin_centers = (edges[:-1] + edges[1:]) / 2

    # manual estimate
    n_max = np.max(nhist)
    n_min = np.min(nhist)
    halfmax = (n_max - n_min)/2
    crossing_ind = np.nonzero(nhist > halfmax)[-1]
    halfwidth = (bin_centers[crossing_ind] +
        (bin_centers[crossing_ind+1] - bin_centers[crossing_ind]) *
        (halfmax - nhist[crossing_ind]) / (nhist[crossing_ind+1] -
         nhist[crossing_ind]))
    fwhm_estimate = 2*halfwidth

    mid = int(round(len(nhist)/2))

    if mode == 1:
        # 68% containment value
        raise RuntimeError('68% mode not implemented yet!')
    elif mode == 2:
        # constant + forward peak
        model = lmfit.models.ConstantModel() + lmfit.models.GaussianModel()
        init_values = {'c': np.min(nhist),
                       'center': 0,
                       'amplitude': np.max(nhist) - np.min(nhist),
                       'sigma': fwhm_estimate / 2.355}
        params = model.make_params(**init_values)
        params['center'].vary = False
        fit = model.fit(nhist, x=bin_centers, params=params)


    elif mode == 3:
        # constant + forward peak + backscatter
        model = (lmfit.models.ConstantModel() +
                 lmfit.models.GaussianModel(prefix='fwd_')) +
                 lmfit.models.GaussianModel(prefix='bk_')) +
        init_values = {'c': np.min(nhist),
                       'fwd_center': 0,
                       'fwd_amplitude': np.max(nhist[:mid]) - np.min(nhist),
                       'fwd_sigma': fwhm_estimate / 2.355,
                       'bk_center': 180,
                       'bk_amplitude': np.max(nhist[mid:]) - np.min(nhist),
                       'bk_sigma': fwhm_estimate / 2.355 * 1.5}
        params = model.make_params(**init_values)
        params['fwd_center'].vary = False
        params['bk_center'].vary = False
        fit = model.fit(nhist, x=bin_centers, params=params)

    return fit

def test_dalpha():
    """
    """
    # test basic scalar-scalar (ints)
    a1 = 5
    a2 = 15
    assert delta_alpha(a1, a2) == 10

    # test basic scalar-scalar (floats)
    a1 = 5.5
    a2 = 15.5
    assert delta_alpha(a1, a2) == 10

    # test wraparound scalar-scalar
    a1 = 175 + 360
    a2 = -175 - 360
    assert delta_alpha(a1, a2) == 10
    a1 = -175 - 360
    a2 = 175 + 360
    assert delta_alpha(a1, a2) == -10

    # test vector-scalar (list)
    a1 = -175
    a2 = [-150, 30, 175]
    assert np.all(delta_alpha(a1, a2) == np.array([25, -155, -10]))

    # test vector-vector (list)
    a1 = [-170, 0, 170]
    a2 = [170.5, 30.5, 150.5]
    assert np.all(delta_alpha(a1, a2) == np.array([-19.5, 30.5, -19.5]))


def test_dbeta():
    """
    """
    print('test_dbeta not implemented yet')


if __name__ == '__main__':
    """
    Run tests.
    """

    test_dalpha()
    test_dbeta()
