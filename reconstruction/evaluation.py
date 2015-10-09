#!/usr/bin/python

import numpy as np
import lmfit
import h5py
import ipdb as pdb

import trackdata


##############################################################################
#                        Algorithm Results class                             #
##############################################################################


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

    def __init__(self, parent=None, filename=None, **kwargs):
        """
        Should be called by a classmethod constructor instead...
        """

        self.parent = parent
        self.filename = filename

        for attr in self.data_attrs():
            if attr in kwargs.keys():
                setattr(self, attr, kwargs[attr])
            else:
                setattr(self, attr, None)

        self.has_alpha = (self.alpha_true_deg is not None)
        self.has_beta = (self.beta_true_deg is not None)

        self.measure_data_length()
        self.input_error_check()

    @classmethod
    def data_attrs(cls):
        """
        List all the data attributes available to the AlgorithmResults class.

        These attributes, if not None, should all be of the same length.
        (i.e. doesn't include parent, filename, has_alpha, has_beta)
        """

        attr_list = (
            'alpha_true_deg',
            'alpha_meas_deg',
            'beta_true_deg',
            'beta_meas_deg',
            'energy_tot_kev',
            'energy_dep_kev',
            'depth_um',
            'is_contained')

        return attr_list

    def measure_data_length(self):
        """
        Data length is taken from the first non-None attribute,
        in order of the data_attrs() list.
        """

        for attr in self.data_attrs():
            if getattr(self, attr) is not None:
                self.data_length = len(getattr(self, attr))
                break
        else:
            raise RuntimeError('AlgorithmResults object requires data')

    def input_error_check(self):
        # type checks
        if (self.parent is not None and
                type(self.parent) is not AlgorithmResults):
            raise RuntimeError(
                'Parent should be an instance of AlgorithmResults')
        if (self.filename is not None and
                type(self.filename) is not str and
                type(self.filename) is not list or
                type(self.filename[0]) is not str):
            raise RuntimeError(
                'Filename should be a string or a list of strings')

        # type conversion
        for attr in self.data_attrs():
            if attr.startswith('is_'):
                setattr(self, attr, bool(getattr(self, attr)))
            else:
                setattr(self, attr, np.array(getattr(self, attr)))

        # related data
        if np.logical_xor(self.alpha_true_deg is None,
                          self.alpha_meas_deg is None):
            raise RuntimeError(
                'Alpha results require both alpha_true and alpha_meas')
        if np.logical_xor(self.beta_true_deg is None,
                          self.beta_meas_deg is None):
            raise RuntimeError(
                'Beta results require both beta_true and beta_meas')

        # data length mismatches
        for attr in self.data_attrs():
            if (getattr(self, attr) is not None and
                    len(getattr(self, attr)) != self.data_length):
                raise RuntimeError(attr + ' length mismatch')

    @classmethod
    def from_h5initial(cls, fieldname, filename=None, h5file=None):
        """
        Construct AlgorithmResults instance from an h5file of MultiAngle
        pixelsize data.

        Inputs:
          fieldname: e.g. 'pix10_5noise0'
          filename: path/name of an HDF5 file
          h5file: the loaded HDF5 file object from h5py.File with read access

        Either filename or h5file must be provided.

        Returns TWO instances of AlgorithmResults: 10.5um, 2.5um.
        """

        if filename is None and h5file is None:
            raise RuntimeError('AlgorithmResults.from_multiangle requires '
                               'either filename or h5file as input')
        if h5file is None:
            h5file = h5py.File(filename, 'r')
        else:
            filename = h5file.filename

        n = 0
        tracks = {'10.5': [[] for _ in len(h5file)],
                  '2.5': [[] for _ in len(h5file)]}

        for evt in h5file:
            if 'Etot' not in evt.attrs or 'Edep' not in evt.attrs:
                continue
            if 'cheat_alpha' not in evt.attrs:
                continue
            if fieldname not in evt:
                continue

            if 'pix10_5noise0' in evt.keys() and 'pix2_5noise0' in evt.keys():
                pix10 = evt['pix10_5noise0']
                pix2 = evt['pix2_5noise0']
                g4track = trackdata.G4Track.from_h5initial(evt)
                tracks['10.5'][n] = trackdata.Track.from_h5initial_one(
                    pix10, g4track)
                tracks['2.5'][n] = trackdata.Track.from_h5initial_one(
                    pix2, g4track)
                n += 1

        results10 = cls.from_track_array(tracks['10.5'])
        results2 = cls.from_track_array(tracks['2.5'])

        return results10, results2

    @classmethod
    def from_track_array(cls, tracks,
                         alg_name='matlab HT v1.5', filename=None):
        """
        Construct AlgorithmResults instance from an array of trackdata.Track
        objects.

        Inputs:
          tracks: list of trackdata.Track objects with algorithm outputs
          alg_name: name of algorithm to take results from
            [default 'matlab HT v1.5']
        """

        alpha_true_deg = np.zeros(len(tracks))
        alpha_meas_deg = np.zeros(len(tracks))
        beta_true_deg = np.zeros(len(tracks))
        beta_meas_deg = np.zeros(len(tracks))
        energy_tot_kev = np.zeros(len(tracks))
        energy_dep_kev = np.zeros(len(tracks))
        depth_um = np.zeros(len(tracks))
        is_contained = np.zeros(len(tracks))

        for i, track in enumerate(tracks):
            alpha_true_deg[i] = track.g4track.alpha_deg
            beta_true_deg[i] = track.g4track.beta_deg
            alpha_meas_deg[i] = track[alg_name].alpha_deg
            beta_meas_deg[i] = track[alg_name].beta_deg
            energy_tot_kev[i] = track.g4track.energy_tot_kev
            energy_dep_kev[i] = track.g4track.energy_dep_kev
            depth_um[i] = track.g4track.depth_um
            is_contained[i] = track.g4track.is_contained

        results = cls(
            alg_name=alg_name, filename=filename,
            alpha_true_deg=alpha_true_deg, alpha_meas_deg=alpha_meas_deg,
            beta_true_deg=beta_true_deg, beta_meas_deg=beta_meas_deg,
            energy_tot_kev=energy_tot_kev, energy_dep_kev=energy_dep_kev,
            depth_um=depth_um, is_contained=is_contained)

        return results

    def select(self, **conditions):
        """
        Construct a new AlgorithmResults object by selecting events out of
        this one.

        New AlgorithmResults has this AlgorithmResults as its .parent

        Input(s):
          **conditions: key-value pairs can include the following:
          beta_min
          beta_max
          energy_min
          energy_max
          depth_min
          depth_max
          is_contained
        """

    def __len__(self):
        """
        length of algorithm results array. For use by len(results)
        """

        return self.data_length

    def __add__(self, new):
        """
        combine two AlgorithmResults objects by concatenating data
        """

        data_attrs =
        for attname in data_attrs:
            if np.logical_xor(getattr(self, attname), getattr(new, attname)):
                raise Warning('asymmetric concatenation of ' + attname)




##############################################################################
#                        Data Selection class                                #
##############################################################################


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
        selection = (np.ones(len(results)) > 0)

        for kw in conditions.keys():
            if kw.starts_with('beta') and not results.has_beta:
                raise RuntimeError(
                    'Cannot select using beta when beta does not exist')
            elif kw.starts_with('energy') and results.energy_tot_kev is None:
                raise RuntimeError(
                    'Cannot select using energy when energy does not exist')
            elif kw.starts_with('depth') and results.depth_um is None:
                raise RuntimeError(
                    'Cannot select using depth when depth does not exist')
            elif kw == 'is_contained' and results.is_contained is None:
                raise RuntimeError(
                    'Cannot select using is_contained when is_contained '
                    'does not exist')

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
        self.length = np.sum(selection)

    def __call__(self):
        """
        If called as a function, return the boolean array of selection.
        """
        return self.selection

    def __len__(self):
        """
        The length of the data selection is the number of True values.
        """

        return self.length


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
    Put all values into (-180, +180]. Operates in-place.
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


##############################################################################
#                        Algorithm Uncertainty class                         #
##############################################################################


class AlgorithmUncertainty(object):
    """
    Produce and store the alpha and beta uncertainty metrics.

    Input: AlgorithmResults object

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

    def __init__(self, alg_results):
        # dalpha=None, beta_true=None, beta_meas=None,
        # mode=2, alpha_mode=None, beta_mode=None):

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
            # TODO: beta


class AlphaUncertainty():
    """
    """
    pass


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


def fit_beta():
    """
    """
    raise NotImplementedError('fit_beta has not been implemented yet')
    pass


##############################################################################
#                                  Testing                                   #
##############################################################################


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

    # TODO
    print('test_dbeta not implemented yet')


def test_alg_results():
    """
    Test AlgorithmResults class.
    """

    # TODO
    pass


def test_alg_uncertainty():
    """
    Test AlgorithmUncertainty class.
    """

    # TODO
    pass


def test_selection():
    """
    Test DataSelection class.
    """

    # TODO
    print('test_selection not implemented yet')


if __name__ == '__main__':
    """
    Run tests.
    """

    test_dalpha()
    test_dbeta()
    test_alg_results()
    test_alg_uncertainty()
    test_selection()
