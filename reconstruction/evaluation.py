#!/usr/bin/python

import numpy as np
import lmfit
import ipdb as pdb


class AlgorithmResults(object):
    """
    Object containing the results of the algorithm on modeled data.

    Contains:
      alpha_true
      alpha_meas
      beta_true
      beta_meas
      Etot
      Edep
      depth
    """

    def __init__(self,
                 alpha_true_deg=None, alpha_meas_deg=None,
                 beta_true_deg=None, beta_meas_deg=None,
                 energy_tot_kev=None, energy_dep_kev=None,
                 depth_um=None, is_contained=None):
        """
        Should be called by a classmethod constructor instead...
        """

        self.alpha_true_deg = alpha_true_deg
        self.alpha_meas_deg = alpha_meas_deg
        self.beta_true_deg = beta_true_deg
        self.beta_meas_deg = beta_meas_deg
        self.energy_tot_kev = energy_tot_kev
        self.energy_dep_kev = energy_dep_kev
        self.depth_um = depth_um
        self.is_contained = is_contained

        self.data_length = len(alpha_true_deg)

    @classmethod
    def from_pixelsize(cls, h5file, fieldname):
        """
        Construct AlgorithmResults instance from an h5file of pixelsize data.
        """

        # . . .

        instance = cls(
            alpha_true_deg=alpha_true_deg, alpha_meas_deg=alpha_meas_deg,
            beta_true_deg=beta_true_deg, beta_meas_deg=beta_meas_deg,
            energy_tot_kev=energy_tot_kev, energy_dep_kev=energy_dep_kev,
            depth_um=depth_um, is_contained=is_contained)

        return instance


class DataSelection(object):
    """
    Object containing data selection for an AlgorithmResults instance.
    """

    def __init__(self, results, **conditions):
        """
        Construct data selection

        Required input:
          results: an AlgorithmResults object

        Optional input(s):
          **conditions: key-value pairs can include the following:
          beta_min
          beta_max
          energy_min
          energy_max
          depth_min
          depth_max
          is_contained
        """

        self.results = results

        # start with all true
        selection = (np.ones(self.data_length) > 0)

        for kw in conditions.keys():
            if kw == 'beta_min':
                param = results.beta_true_deg
                comparator = np.greater
            elif kw == 'beta_max':
                param = results.beta_true_deg
                comparator = np.less
            elif kw == 'energy_min':
                param = results.energy_tot_kev
                comparator = np.greater
            elif kw == 'energy_max':
                param = results.energy_tot_kev
                comparator = np.less
            elif kw == 'depth_min':
                param = results.depth_um
                comparator = np.greater
            elif kw == 'depth_max':
                param = results.depth_um
                comparator = np.less
            elif kw == 'is_contained':
                param = results.is_contained
                comparator = np.equal
            else:
                raise RuntimeError(
                    'Condition keyword not found: {}'.format(kw))

            selection = np.logical_and(
                selection, comparator(param, conditions[kw]))
            self.selection = selection

    # def __call__(self):


def delta_alpha(alpha_true_deg, alpha_meas_deg):
    """
    Compute alpha_meas - alpha_true, returning value on (-180, +180].

    scalar, scalar: return a scalar
    vector, scalar: return a vector (each vector value compared to scalar)
    vector, vector: vectors should be same size. compare elementwise.
    """

    # type conversion... also copies the data to avoid modifying the inputs
    alpha_true_deg = np.array(alpha_true_deg)
    alpha_meas_deg = np.array(alpha_meas_deg)

    dalpha = alpha_meas_deg - alpha_true_deg
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
            raise RuntimeError('AlgorithmUncertainty requires either' +
                               ' dalpha, or both beta_true and beta_meas')
        elif np.logical_xor(beta_true is None, beta_meas is None):
            raise RuntimeError('Both beta_true and beta_meas must be provided')
        if alpha_mode is None:
            alpha_mode = mode
        if beta_mode is None:
            beta_mode = mode

        if dalpha is not None:
            alpha_result = fit_alpha(dalpha, mode=alpha_mode)
            # TODO: make this more robust and organized
            self.a_FWHM = alpha_result.params['fwhm'].value
            self.a_f = alpha_result.params['f'].value
            self.a_frandom = alpha_result.params['f_random'].value

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
    dalpha = np.abs(dalpha.flatten())

    n_values = len(dalpha)
    resolution = np.minimum(100.0 * 180.0 / n_values, 15)   # from MATLAB
    n_bins = np.ceil(180 / resolution)
    nhist, edges = np.histogram(dalpha, bins=n_bins, range=(0.0, 180.0))
    bin_centers = (edges[:-1] + edges[1:]) / 2

    # manual estimate
    n_max = np.max(nhist)
    n_min = np.min(nhist)
    halfmax = (n_max - n_min)/2
    crossing_ind = np.nonzero(nhist > halfmax)[0][-1]
    halfwidth = (
        bin_centers[crossing_ind] +
        (bin_centers[crossing_ind + 1] - bin_centers[crossing_ind]) *
        (halfmax - nhist[crossing_ind]) / (nhist[crossing_ind + 1] -
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
        # TODO: make this more robust
        peak_fraction = (
            fit.params['amplitude'].value / 2 / resolution / n_values)
        fit.params.add('f', vary=False, value=peak_fraction)
        random_fraction = fit.params['c'].value * 180 / resolution / n_values
        fit.params.add('f_random', vary=False, value=random_fraction)

    elif mode == 3:
        # constant + forward peak + backscatter
        model = (lmfit.models.ConstantModel() +
                 lmfit.models.GaussianModel(prefix='fwd_') +
                 lmfit.models.GaussianModel(prefix='bk_'))
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

    return None


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
