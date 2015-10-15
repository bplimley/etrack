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
                type(self.parent) is not AlgorithmResults and
                (type(self.parent) is not list or
                 type(self.parent[0]) is not AlgorithmResults)):
            raise RuntimeError(
                'Parent should be an instance of AlgorithmResults')
        if (self.filename is not None and
                type(self.filename) is not str and
                (type(self.filename) is not list or
                 type(self.filename[0]) is not str)):
            raise RuntimeError(
                'Filename should be a string or a list of strings')

        # type conversion
        for attr in self.data_attrs():
            if getattr(self, attr) is not None:
                if attr.startswith('is_'):
                    setattr(self, attr, getattr(self, attr).astype(bool))
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
        tracks = {'10.5': np.empty(len(h5file)),
                  '2.5': np.empty(len(h5file))}

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
          beta_min (alias for beta_true_min)
          beta_max (alias for beta_true_max)
          beta_meas_min
          beta_meas_max
          energy_min
          energy_max
          depth_min
          depth_max
          is_contained
        """

        # start with all true
        selection = (np.ones(len(self)) > 0)

        for kw in conditions.keys():
            if kw.lower().startswith('beta') and not self.has_beta:
                raise RuntimeError(
                    'Cannot select using beta when beta does not exist')
            elif (kw.lower().startswith('energy') and
                    self.energy_tot_kev is None):
                raise RuntimeError(
                    'Cannot select using energy when energy does not exist')
            elif kw.lower().startswith('depth') and self.depth_um is None:
                raise RuntimeError(
                    'Cannot select using depth when depth does not exist')
            elif kw.lower() == 'is_contained' and self.is_contained is None:
                raise RuntimeError(
                    'Cannot select using is_contained when is_contained '
                    'does not exist')

            # by default, do not wrap the parameter in a function.
            wrapper = lambda x: x
            if kw.lower() == 'beta_min' or kw.lower() == 'beta_true_min':
                param = self.beta_true_deg
                wrapper = np.abs
                comparator = np.greater
            elif kw.lower() == 'beta_max' or kw.lower() == 'beta_true_max':
                param = self.beta_true_deg
                wrapper = np.abs
                comparator = np.less
            elif kw.lower() == 'beta_meas_min':
                param = self.beta_meas_deg
                comparator = np.greater
            elif kw.lower() == 'beta_meas_max':
                param = self.beta_meas_deg
                comparator = np.less
            elif kw.lower() == 'energy_min':
                param = self.energy_tot_kev
                comparator = np.greater
            elif kw.lower() == 'energy_max':
                param = self.energy_tot_kev
                comparator = np.less
            elif kw.lower() == 'depth_min':
                param = self.depth_um
                comparator = np.greater
            elif kw.lower() == 'depth_max':
                param = self.depth_um
                comparator = np.less
            elif kw.lower() == 'is_contained':
                param = self.is_contained
                comparator = np.equal
            else:
                raise RuntimeError(
                    'Condition keyword not found: {}'.format(kw))

            selection = np.logical_and(
                selection, comparator(wrapper(param), conditions[kw.lower()]))

        selected_data = dict()
        for attr in self.data_attrs():
            if getattr(self, attr) is None:
                selected_data[attr] = None
            else:
                selected_data[attr] = getattr(self, attr)[selection]

        return AlgorithmResults(parent=self,
                                filename=self.filename,
                                **selected_data)

    def __len__(self):
        """
        length of algorithm results array. For use by len(results)
        """

        return self.data_length

    def __add__(self, added):
        """
        combine two AlgorithmResults objects by concatenating data
        """

        # if, say, one object has an is_contained record and the other
        #   doesn't (is_contained = None), then populate the nonexistent data
        #   record with np.nan's, I guess. But issue a warning.

        new = dict()
        for attname in self.data_attrs():
            data1 = getattr(self, attname)
            data2 = getattr(added, attname)
            if data1 is not None and data2 is not None:
                new[attname] = np.concatenate((data1, data2))
            elif data1 is None and data2 is None:
                new[attname] = None
            elif data1 is not None and data2 is None:
                temp = np.array([np.nan for _ in range(len(added))])
                new[attname] = np.concatenate((data1, temp))
                raise Warning('asymmetric concatenation of ' + attname)
            elif data1 is None and data2 is not None:
                temp = np.array([np.nan for _ in range(len(self))])
                new[attname] = np.concatenate((temp, data2))
                raise Warning('asymmetric concatenation of ' + attname)

        # non-data attributes
        if self.parent is not None or added.parent is not None:
            new_parent = [self.parent, added.parent]
        else:
            new_parent = None

        if self.filename is not None or added.filename is not None:
            new_filename = [self.filename, added.filename]
        else:
            new_filename = None

        return AlgorithmResults(parent=new_parent,
                                filename=new_filename,
                                **new)


##############################################################################
#                           Other misc functions                             #
##############################################################################


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
#                        Algorithm Uncertainty classes                       #
##############################################################################


class Uncertainty(object):
    """
    Either an alpha uncertainty or beta uncertainty object.
    """

    def __init__(self, alg_results):
        """
        Initialization is common to all Uncertainty objects.

        The methods will be overwritten in subclasses.
        """

        self.compute_delta(alg_results)

        self.n_values = len(self.delta)
        self.prepare_data()
        self.perform_fit()
        self.compute_metrics()

    def compute_delta(self, alg_results):
        self.delta = []
        pass

    def prepare_data(self):
        pass

    def perform_fit(self):
        pass

    def compute_metrics(self):
        pass


class AlphaUncertainty(Uncertainty):
    """
    An alpha uncertainty calculation method
    """

    angle_type = 'alpha'

    def compute_delta(self, alg_results):
        dalpha = delta_alpha(alg_results.alpha_true_deg,
                             alg_results.alpha_meas_deg)
        adjust_dalpha(dalpha)
        self.delta = np.abs(dalpha.flatten())


class BetaUncertainty(Uncertainty):
    """
    A beta uncertainty calculation method
    """

    angle_type = 'beta'

    def compute_delta(self, alg_results):
        self.delta = delta_beta(alg_results.beta_true_deg,
                                alg_results.beta_meas_deg)


class UncertaintyParameter(object):
    """
    One metric parameter of uncertainty.

    Attributes:
      name (str): name of this parameter, e.g. "FWHM"
      fit_name (str): name of the fit, e.g. "GaussPlusConstant"
      value (float): value from the fit, e.g. 29.8
      uncertainty (float, float): tuple of (lower, upper) 1-sigma uncertainty
      units (str): units of the value and uncertainty, e.g. "degrees" "\cir" ?
      axis_min (float): lower edge of axis on a plot, e.g. 0
      axis_max (float): upper edge of axis on a plot, e.g. 120
    """

    def __init__(self, name=None, fit_name=None,
                 value=None, uncertainty=None, units=None,
                 axis_min=None, axis_max=None):
        """
        """

        self.name = name
        self.fit_name = fit_name
        self.value = value
        self.uncertainty = uncertainty
        self.units = units
        self.axis_min = axis_min
        self.axis_max = axis_max


class AlgorithmUncertainty(object):
    """
    Produce and store the alpha and beta uncertainty metrics.

    Input: AlgorithmResults object
    """
    # ...
    # mode: sets both alpha_mode and beta_mode simultaneously.
    #   (default: mode=2)
    #
    # alpha_mode:
    #   1: 68% metrics (not coded yet)
    #   2: forward peak, constant (random) background
    #   3: forward peak, backscatter peak, constant background
    #
    # beta_mode:
    #   1: 68% containment
    #   2: RMS
    #   3: zero fraction, ???

    def __init__(self, alg_results, aunc=None, bunc=None):
        """
        Initialize from alg_results object.

        Compute alpha uncertainties (if alpha data available)
        and beta uncertainties (if beta data available)
        using the fit classes, aunc and bunc
        """

        # TODO: default fit algorithms

        has_alpha = alg_results.has_alpha
        has_beta = alg_results.has_beta

        if not has_alpha and not has_beta:
            raise RuntimeError(
                'AlgorithmUncertainty requires either alpha or beta')
        if has_alpha:
            self.alpha_result = aunc(alg_results)
        if has_beta:
            self.beta_result = bunc(alg_results)

        # if has_alpha:
        #     alpha_result = fit_alpha(dalpha, mode=alpha_mode)
        #     self.a_FWHM = alpha_result.params['fwhm'].value
        #     self.a_f = alpha_result.params['f'].value
        #     self.a_frandom = alpha_result.params['f_random'].value
        #
        # if has_beta:
        #     beta_result = fit_beta(beta_true=beta_true, beta_meas=beta_meas,
        #                            mode=beta_mode)


class AlphaGaussPlusConstant(AlphaUncertainty):
    """
    Fitting d-alpha distribution with gaussian plus constant
    """

    def prepare_data(self):
        """
        """

        # resolution calculation is from MATLAB
        self.resolution = np.minimum(100.0 * 180.0 / self.n_values, 15)
        n_bins = np.ceil(180 / self.resolution)
        nhist, edges = np.histogram(
            self.delta, bins=n_bins, range=(0.0, 180.0))

        self.nhist = nhist
        self.xhist = (edges[:-1] + edges[1:]) / 2

    def perform_fit(self):
        """
        """

        # manual estimate
        halfmax = self.nhist.ptp()/2
        crossing_ind = np.nonzero(self.nhist > halfmax)[0][-1]
        halfwidth = (
            self.xhist[crossing_ind] +
            (self.xhist[crossing_ind + 1] - self.xhist[crossing_ind]) *
            (halfmax - self.nhist[crossing_ind]) /
            (self.nhist[crossing_ind + 1] - self.nhist[crossing_ind]))
        fwhm_estimate = 2 * halfwidth

        # mid = int(round(len(self.nhist)/2))

        # constant + forward peak
        model = lmfit.models.ConstantModel() + lmfit.models.GaussianModel()
        # How this is working:
        #   res and n are defined as parameters, for use in expressions.
        #   f is defined from amplitude.
        #   f_random is constrained to be 1 - f.
        #   c is set not to be varied, but to depend on f_random.
        #   center is fixed to 0.
        #   (FWHM is already defined for the Gaussian.)
        init_values = {'c': self.nhist.min(),
                       'center': 0,
                       'amplitude': self.nhist.ptp(),
                       'sigma': fwhm_estimate / 2.355}
        params = model.make_params(**init_values)
        params.add('res', vary=False, value=self.resolution)
        params.add('n', vary=False, value=self.n_values)
        params.add('f', vary=False, expr='amplitude / 2 / res / n')
        params.add('f_random', vary=False, expr='1-f')
        params['c'].set(value=None, vary=False, expr='res*n*f_random/180')
        params['center'].vary = False
        self.fit = model.fit(self.nhist, x=self.xhist, params=params)

        if not self.fit.success:
            raise RuntimeError('Fit failed!')

    def compute_metrics(self):
        """
        """

        if not self.fit.errorbars:
            fwhm_unc = np.nan
            f_unc = np.nan
            raise Warning('Could not compute errorbars in fit')
        else:
            fwhm_unc = self.fit.params['fwhm'].stderr
            f_unc = self.fit.params['f'].stderr

        fwhm_param = UncertaintyParameter(
            name='FWHM',
            fit_name=self.__class__,
            value=self.fit.params['fwhm'].value,
            uncertainty=(fwhm_unc, fwhm_unc),
            units='degrees',
            axis_min=0.0,
            axis_max=120.0)
        f_param = UncertaintyParameter(
            name='peak fraction',
            fit_name=self.__class__,
            value=self.fit.params['f'].value,
            uncertainty=(f_unc, f_unc),
            units='%',
            axis_min=0.0,
            axis_max=100.0)
        self.metrics = {'FWHM': fwhm_param,
                        'f': f_param}


class AlphaGaussPlusConstantPlusBackscatter(AlphaUncertainty):
    """
    Fitting d-alpha distribution with gaussian + constant + backscatter
    """

    def prepare_data(self):
        """
        """

        # resolution calculation is from MATLAB
        self.resolution = np.minimum(100.0 * 180.0 / self.n_values, 15)
        n_bins = np.ceil(180 / self.resolution)
        nhist, edges = np.histogram(
            self.delta, bins=n_bins, range=(0.0, 180.0))

        self.nhist = nhist
        self.xhist = (edges[:-1] + edges[1:]) / 2

    def perform_fit(self):
        """
        """

        # manual estimate
        halfmax = self.nhist.ptp()/2
        crossing_ind = np.nonzero(self.nhist > halfmax)[0][-1]
        halfwidth = (
            self.xhist[crossing_ind] +
            (self.xhist[crossing_ind + 1] - self.xhist[crossing_ind]) *
            (halfmax - self.nhist[crossing_ind]) /
            (self.nhist[crossing_ind + 1] - self.nhist[crossing_ind]))
        fwhm_estimate = 2 * halfwidth

        mid = int(round(len(self.nhist)/2))

        # constant + forward peak
        model = (lmfit.models.ConstantModel() +
                 lmfit.models.GaussianModel(prefix='fwd') +
                 lmfit.models.GaussianModel(prefix='bk'))
        # How this is working:
        #   res and n are defined as parameters, for use in expressions.
        #   f is defined from amplitude.
        #   f_random is constrained to be 1 - f.
        #   c is set not to be varied, but to depend on f_random.
        #   center is fixed to 0.
        #   (FWHM is already defined for the Gaussian.)
        init_values = {'c': self.nhist.min(),
                       'fwd_center': 0,
                       'fwd_amplitude': self.nhist[:mid].ptp(),
                       'fwd_sigma': fwhm_estimate / 2.355,
                       'bk_center': 180,
                       'bk_amplitude': self.nhist[mid:].ptp(),
                       'bk_sigma': fwhm_estimate / 2.355 * 1.5}
        params = model.make_params(**init_values)
        params.add('res', vary=False, value=self.resolution)
        params.add('n', vary=False, value=self.n_values)
        params.add('f', vary=False, expr='fwd_amplitude / 2 / res / n')
        params.add('f_bk', vary=False, expr='bk_amplitude / 2 / res / n')
        params.add('f_random', vary=False, expr='1-f-f_bk')
        params['c'].set(value=None, vary=False, expr='res*n*f_random/180')
        params['fwd_center'].vary = False
        params['bk_center'].vary = False
        self.fit = model.fit(self.nhist, x=self.xhist, params=params)

    def compute_metrics(self):
        """
        """

        if not self.fit.errorbars:
            fwhm_unc = np.nan
            f_unc = np.nan
            f_bk_unc = np.nan
            raise Warning('Could not compute errorbars in fit')
        else:
            fwhm_unc = self.fit.params['fwhm'].stderr
            f_unc = self.fit.params['f'].stderr
            f_bk_unc = self.fit.params['f_bk'].stderr

        fwhm_param = UncertaintyParameter(
            name='FWHM',
            fit_name=self.__class__,
            value=self.fit.params['fwhm'].value,
            uncertainty=(fwhm_unc, fwhm_unc),
            units='degrees',
            axis_min=0.0,
            axis_max=120.0)
        f_param = UncertaintyParameter(
            name='peak fraction',
            fit_name=self.__class__,
            value=self.fit.params['f'].value,
            uncertainty=(f_unc, f_unc),
            units='%',
            axis_min=0.0,
            axis_max=100.0)
        f_bk_param = UncertaintyParameter(
            name='backscatter fraction',
            fit_name=self.__class__,
            value=self.fit.params['f_bk'].value,
            uncertainty=(f_bk_unc, f_bk_unc),
            units='%',
            axis_min=0.0,
            axis_max=100.0)
        self.metrics = {'FWHM': fwhm_param,
                        'f': f_param,
                        'f_bk': f_bk_param}


class Alpha68(AlphaUncertainty):
    """
    68% containment value for alpha.
    """

    def prepare_data(self):
        """
        Calculate threshold values for 68% and +/- 1sigma
        """

        # see CalculateUncertainties.m from 2010
        resolution = 0.01
        n_bins = np.ceil(180 / resolution)
        f = 0.68

        self.n_thresh = self.n_values * f

        n_sigma = np.sqrt(self.n_values * f * (1 - f))

        self.n_leftsigma = self.n_thresh - n_sigma
        self.n_rightsigma = self.n_thresh + n_sigma

        nhist, edges = np.histogram(
            self.delta, bins=n_bins, range=(0.0, 180.0))

        self.nhist = nhist
        self.xhist = edges[1:]      # right edge of each bin (conservative)

    def perform_fit(self):
        """
        """

        n = self.nhist.cumsum()

        c68 = np.flatnonzero(n > self.n_thresh)[0]
        self.contains68_value = self.xhist[c68]

        c68_left = np.flatnonzero(n > self.n_leftsigma)[0]
        c68_lower = c68 - c68_left
        self.contains68_lower = self.xhist[c68_lower]

        c68_right = np.flatnonzero(n > self.n_rightsigma)[0]
        c68_upper = c68_right - c68
        self.contains68_upper = self.xhist[c68_upper]

    def compute_metrics(self):
        """
        """

        value = self.contains68_value
        lower = self.contains68_lower
        upper = self.contains68_upper
        contains68_param = UncertaintyParameter(
            name='sigma_tilde',
            fit_name=self.__class__,
            value=value,
            uncertainty=(lower, upper),
            units='degrees',
            axis_min=0.0,
            axis_max=130.0)

        self.metrics = {'contains68': contains68_param}


class BetaRms(BetaUncertainty):
    """
    Calculate RMS for all values of delta beta.
    """

    def prepare_data(self):
        pass

    def perform_fit(self):
        self.rms = np.sqrt(np.mean(np.square(self.delta)))

    def compute_metrics(self):
        rms_unc = self.rms / np.sqrt(2*(self.n_values - 1))
        rms_param = UncertaintyParameter(
            name='RMS',
            fit_name=self.__class__,
            value=self.rms,
            uncertainty=(rms_unc, rms_unc),
            units='degrees',
            axis_min=0.0,
            axis_max=40.0)

        self.metrics = {'RMS': rms_param}


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

# AlgorithmResults class


def generate_random_alg_results(
        length=100,
        filename=None, parent=None,
        has_alpha=True, has_beta=True, has_contained=True,
        a_fwhm=30, a_f=0.7, b_rms=20, b_f=0.5):
    """
    Generate an AlgorithmResults instance for testing.

    Data values are pseudo-realistic. But:
      -there is no energy dependence
      -all tracks are contained and Edep == Etot
    """

    if has_alpha:
        alpha_true = np.random.uniform(low=-180, high=180, size=length)
        good_alpha_length = np.round(a_f * length)
        alpha_meas_bad = np.random.uniform(
            low=-180, high=180, size=(length - good_alpha_length))
        alpha_meas_good = alpha_true[:good_alpha_length] + np.random.normal(
            loc=0.0, scale=(a_fwhm)/2.355, size=good_alpha_length)
        alpha_meas = np.concatenate((alpha_meas_good, alpha_meas_bad))
    else:
        alpha_true = None
        alpha_meas = None
    if has_beta:
        beta_true = np.random.triangular(
            left=-90, mode=0, right=90, size=length)
        good_beta_length = np.round(b_f * length)
        beta_meas_bad = np.zeros(length - good_beta_length)
        beta_meas_good = beta_true[:good_beta_length] + np.random.normal(
            loc=0.0, scale=b_rms, size=good_beta_length)
        redo = np.logical_or(beta_meas_good > 90, beta_meas_good < -90)
        beta_meas_good[redo] = np.random.uniform(
            low=-90, high=90, size=np.sum(redo))
        beta_meas = np.concatenate((beta_meas_good, beta_meas_bad))
    else:
        beta_true = None
        beta_meas = None

    energy_tot_kev = np.random.uniform(low=100, high=478, size=length)
    energy_dep_kev = energy_tot_kev
    depth = np.random.uniform(low=0, high=650, size=length)
    is_contained = (np.ones(length) > 0)

    return AlgorithmResults(
        parent=parent, filename=filename,
        alpha_true_deg=alpha_true, alpha_meas_deg=alpha_meas,
        beta_true_deg=beta_true, beta_meas_deg=beta_meas,
        energy_tot_kev=energy_tot_kev, energy_dep_kev=energy_dep_kev,
        depth_um=depth, is_contained=is_contained)


def generate_hist_from_results(alg_results, resolution=1.0):
    """
    Generate nhist, xhist from the alpha data in an alg_results instance.

    (Might be superseded by methods in the AlphaUncertainty classes...)
    """

    dalpha = delta_alpha(alg_results.alpha_true_deg,
                         alg_results.alpha_meas_deg)
    adjust_dalpha(dalpha)
    dalpha = np.abs(dalpha.flatten())

    n_bins = np.ceil(180 / resolution)
    nhist, edges = np.histogram(
        dalpha, bins=n_bins, range=(0.0, 180.0))

    nhist = nhist
    xhist = (edges[:-1] + edges[1:]) / 2

    return nhist, xhist


def test_alg_results():
    """
    Test AlgorithmResults class.
    """

    # basic
    generate_random_alg_results()
    generate_random_alg_results(has_beta=False, has_contained=False)

    # length
    assert len(generate_random_alg_results(length=100)) == 100
    assert len(generate_random_alg_results(has_alpha=False, length=100)) == 100

    # subtests
    test_alg_results_input_check()
    test_alg_results_add()
    test_alg_results_select()


def test_alg_results_input_check():
    """
    Test AlgorithmResults input check
    """

    # errors
    try:
        AlgorithmResults(filename=5,
                         alpha_true_deg=[10, 20],
                         alpha_meas_deg=[20, 25])
    except RuntimeError:
        pass
    else:
        print('Failed to catch AlgorithmResults filename error')

    try:
        AlgorithmResults(alpha_true_deg=np.random.random(30),
                         alpha_meas_deg=np.random.random(29))
    except RuntimeError:
        pass
    else:
        print('Failed to catch data length mismatch')

    try:
        AlgorithmResults(alpha_true_deg=np.random.random(30))
    except RuntimeError:
        pass
    else:
        print('Failed to catch missing alpha_meas_deg')


def test_alg_results_add():
    """
    Test AlgorithmResults class.
    """

    len1 = 1000
    len2 = 100
    # basic
    x = generate_random_alg_results(length=len1)
    y = generate_random_alg_results(length=len2)
    z = x + y
    assert len(z) == len1 + len2
    assert z.filename is None
    assert z.parent is None

    # symmetric None's
    x = generate_random_alg_results(length=len1, has_beta=False)
    y = generate_random_alg_results(length=len2, has_beta=False)
    z = x + y
    assert z.has_beta is False
    assert z.beta_true_deg is None

    # asymmetric None's
    try:
        x = generate_random_alg_results(length=len1, has_beta=False)
        y = generate_random_alg_results(length=len2, has_beta=True)
        z = x + y
        assert z.has_beta is True
        assert np.sum(np.isnan(z.beta_true_deg)) == len1
    except Warning:
        pass
    else:
        print('Failed to warn on asymmetric concatenation (None + data)')
    try:
        x = generate_random_alg_results(length=len1, has_beta=True)
        y = generate_random_alg_results(length=len2, has_beta=False)
        z = x + y
        assert z.has_beta is True
        assert np.sum(np.isnan(z.beta_true_deg)) == len2
    except Warning:
        pass
    else:
        print('Failed to warn on asymmetric concatenation (data + None)')

    # non-data attributes
    x = generate_random_alg_results(filename='asdf', length=len1)
    y = generate_random_alg_results(filename='qwerty', length=len2)
    z = x + y
    assert len(z.filename) == 2
    assert z.filename[0] == 'asdf'
    assert z.filename[1] == 'qwerty'
    # also testing the 'parent' property of AlgorithmResults here
    xx = generate_random_alg_results(parent=x, filename='asdf', length=len1)
    yy = generate_random_alg_results(parent=y, filename='qwerty', length=len2)
    zz = xx + yy
    assert len(zz.filename) == 2
    assert zz.filename[0] == 'asdf'
    assert zz.filename[1] == 'qwerty'
    assert len(zz.parent) == 2
    assert zz.parent[0] is x
    assert zz.parent[1] is y


def test_alg_results_select():
    """
    Test AlgorithmResults class.
    """

    len1 = 1000

    # basic
    x = generate_random_alg_results(length=len1)
    x = generate_random_alg_results(length=len1, has_beta=False)
    y = x.select(energy_min=300)
    assert type(y) is AlgorithmResults
    x.select(energy_min=300, energy_max=400, depth_min=200)
    x.select(is_contained=True, depth_min=200)

    # handle all-false result
    y = x.select(energy_min=300, energy_max=300, depth_min=200, depth_max=200)
    assert len(y) == 0
    assert not y.has_beta

    # input error checking
    x = generate_random_alg_results(length=len1, has_beta=False)
    try:
        x.select(beta_min=20)
    except RuntimeError:
        pass
    else:
        print('AlgorithmResults.select() failed to raise error with no beta')

    try:
        x.select(asdf_max=500)
    except RuntimeError:
        pass
    else:
        print('AlgorithmResults.select() failed to ' +
              'raise error on bad condition')


def test_alg_uncertainty():
    """
    Test AlgorithmUncertainty class.
    """

    # TODO
    pass


if __name__ == '__main__':
    """
    Run tests.
    """

    # test_dalpha()
    # test_dbeta()
    test_alg_results()
    # test_alg_uncertainty()
