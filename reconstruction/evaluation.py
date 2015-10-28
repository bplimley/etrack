#!/usr/bin/python

import numpy as np
import lmfit
import h5py
import ipdb as pdb

import trackdata
from trackdata import ClassAttr


##############################################################################
#                        Algorithm Results class                             #
##############################################################################


class AlgorithmResults(object):
    """
    Object containing the results of the algorithm on modeled data.

    Data attributes:
      alpha_true
      alpha_meas
      beta_true
      beta_meas
      Etot
      Edep
      depth

    Other attributes:
      has_alpha
      has_beta
      parent
      filename
      uncertainty_list
      alpha_unc (shortcut to first alpha uncertainty object)
      beta_unc (shortcut to first beta uncertainty object)
    """

    class_name = 'AlgorithmResults'

    def __init__(self, parent=None, filename=None, **kwargs):
        """
        Should be called by a classmethod constructor instead...
        """

        # 'parent' and 'filename' will be converted to lists if they are not
        #   None. this is performed in input_error_check()
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

        self.uncertainty_list = []
        self.alpha_unc = None
        self.beta_unc = None

        self.get_data_format()

    def get_data_format(self):
        """
        Data format for writing to HDF5 (see trackdata.py)
        """
        self.data_format = (
            ClassAttr('parent', AlgorithmResults,
                      may_be_none=True, is_user_object=True,
                      is_always_list=True),
            ClassAttr('filename', str,
                      may_be_none=True, is_always_list=True),
            ClassAttr('has_alpha', bool),
            ClassAttr('has_beta', bool),
            ClassAttr('data_length', int),
            ClassAttr('uncertainty_list', Uncertainty,
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
            raise InputError('AlgorithmResults object requires data')

    def input_error_check(self):
        # type checks
        if (self.parent is not None and
                type(self.parent) is not AlgorithmResults and
                (type(self.parent) is not list or
                 type(self.parent[0]) is not AlgorithmResults)):
            raise InputError(
                'Parent should be an instance of AlgorithmResults')
        if (self.filename is not None and
                type(self.filename) is not str and
                (type(self.filename) is not list or
                 type(self.filename[0]) is not str)):
            raise InputError(
                'Filename should be a string or a list of strings')

        # type conversion
        # parent and filename should always be lists, if they are not None,
        #   even if only 1 element
        if type(self.parent) is AlgorithmResults:
            self.parent = [self.parent]
        if type(self.filename) is str:
            self.filename = [self.filename]
        for attr in self.data_attrs():
            if getattr(self, attr) is not None:
                if attr.startswith('is_'):
                    setattr(self, attr, getattr(self, attr).astype(bool))
                else:
                    setattr(self, attr, np.array(getattr(self, attr)))

        # related data
        if np.logical_xor(self.alpha_true_deg is None,
                          self.alpha_meas_deg is None):
            raise InputError(
                'Alpha results require both alpha_true and alpha_meas')
        if np.logical_xor(self.beta_true_deg is None,
                          self.beta_meas_deg is None):
            raise InputError(
                'Beta results require both beta_true and beta_meas')

        # data length mismatches
        for attr in self.data_attrs():
            if (getattr(self, attr) is not None and
                    len(getattr(self, attr)) != self.data_length):
                raise InputError(attr + ' length mismatch')

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
            raise InputError('AlgorithmResults.from_multiangle requires '
                             'either filename or h5file as input')
        if h5file is None:
            h5file = h5py.File(filename, 'r')
        else:
            filename = h5file.filename

        n = 0
        tracks = [[] for i in range(len(h5file))]

        for evt in h5file.itervalues():
            if 'Etot' not in evt.attrs or 'Edep' not in evt.attrs:
                continue
            if 'cheat_alpha' not in evt.attrs:
                continue
            if fieldname not in evt:
                continue

            g4track = trackdata.G4Track.from_h5initial(evt)
            tracks[n] = trackdata.Track.from_h5initial_one(
                evt[fieldname], g4track)
            n += 1

        tracks = tracks[:n]
        return cls.from_track_array(tracks)

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
            if alg_name in track.algorithms:
                alpha_meas_deg[i] = track[alg_name].alpha_deg
                beta_meas_deg[i] = track[alg_name].beta_deg
            else:
                alpha_meas_deg[i] = np.nan
                beta_meas_deg[i] = np.nan
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
                raise SelectionError(
                    'Cannot select using beta when beta does not exist')
            elif (kw.lower().startswith('energy') and
                    self.energy_tot_kev is None):
                raise SelectionError(
                    'Cannot select using energy when energy does not exist')
            elif kw.lower().startswith('depth') and self.depth_um is None:
                raise SelectionError(
                    'Cannot select using depth when depth does not exist')
            elif kw.lower() == 'is_contained' and self.is_contained is None:
                raise SelectionError(
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
                raise SelectionError(
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

    def add_uncertainty(self, uncertainty_class):
        """
        Attach a new uncertainty object onto this results object.
        """

        if not np.any(
                [u.angle_type == uncertainty_class.angle_type
                 for u in self.uncertainty_list]):
            first = True
        else:
            first = False

        self.uncertainty_list.append(uncertainty_class(self))

        if first:
            # make shortcut, like
            #  alg_results.alpha_unc = alg_results.uncertainty_list[0]
            attr = uncertainty_class.angle_type + '_unc'
            setattr(self, attr, self.uncertainty_list[-1])

    def add_default_uncertainties(self):
        """
        add_uncertainty with default alpha and default beta.

        Defaults are defined at the bottom of the Algorithm Uncertainty Classes
            section of evaluation.py.
        """

        self.add_uncertainty(DefaultAlphaUncertainty)
        self.add_uncertainty(DefaultBetaUncertainty)

    def list_uncertainties(self, angle_type=None):
        """
        List the names of all uncertainty objects attached to this
        results object.
        """

        if angle_type is None:
            return_list = [u.name for u in self.uncertainty_list]
        else:
            return_list = [u.name for u in self.uncertainty_list
                           if u.angle_type == angle_type]

        return return_list

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
                raise DataWarning('asymmetric concatenation of ' + attname)
            elif data1 is None and data2 is not None:
                temp = np.array([np.nan for _ in range(len(self))])
                new[attname] = np.concatenate((temp, data2))
                raise DataWarning('asymmetric concatenation of ' + attname)

        # non-data attributes:
        # both parent and filename should always be lists if they are not None
        if isinstance(self.parent, list) and isinstance(added.parent, list):
            new_parent = self.parent[:]
            new_parent.extend(added.parent)
        elif isinstance(self.parent, list):
            new_parent = self.parent[:]
        elif isinstance(added.parent, list):
            new_parent = added.parent[:]
        else:
            new_parent = None

        if (isinstance(self.filename, list) and
                isinstance(added.filename, list)):
            new_filename = self.filename[:]
            new_filename.extend(added.filename)
        elif isinstance(self.filename, list):
            new_filename = self.filename[:]
        elif isinstance(added.filename, list):
            new_filename = added.filename[:]
        else:
            new_filename = None

        return AlgorithmResults(parent=new_parent,
                                filename=new_filename,
                                **new)


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

        self.n_values = np.sum(np.logical_and(
            np.logical_not(np.isnan(self.delta)),
            np.logical_not(np.isinf(self.delta))))
        self.prepare_data()
        self.perform_fit()
        self.compute_metrics()

        self.get_data_format()

    def get_data_format(self):
        """
        Data format for writing to HDF5 (see trackdata.py)
        """
        self.data_format = (
            ClassAttr('delta', np.ndarray, make_dset=True),
            ClassAttr('n_values', int),
            ClassAttr('metrics', UncertaintyParameter,
                      is_always_dict=True, is_user_object=True),
            ClassAttr('angle_type', str),
        )

    def compute_delta(self, alg_results):
        self.delta = []
        pass

    def prepare_data(self):
        pass

    def perform_fit(self):
        pass

    def compute_metrics(self):
        pass

    @classmethod
    def classname_extract(cls, obj):
        """
        Take an object and extract the class name from __class__.
        """

        full_string = str(obj.__class__)
        classname = full_string.split('.')[-1].split("'")[0]

        return classname


class AlphaUncertainty(Uncertainty):
    """
    An alpha uncertainty calculation method
    """

    angle_type = 'alpha'

    def compute_delta(self, alg_results):
        dalpha = self.delta_alpha(alg_results.alpha_true_deg,
                                  alg_results.alpha_meas_deg)
        self.delta = np.abs(dalpha.flatten())

    @classmethod
    def delta_alpha(cls, alpha_true_deg, alpha_meas_deg):
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
        dalpha = cls.adjust_dalpha(dalpha)

        return dalpha

    @classmethod
    def adjust_dalpha(cls, dalpha):
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

        return dalpha


class BetaUncertainty(Uncertainty):
    """
    A beta uncertainty calculation method
    """

    angle_type = 'beta'

    def compute_delta(self, alg_results):
        self.delta = self.delta_beta(alg_results.beta_true_deg,
                                     alg_results.beta_meas_deg)

    @classmethod
    def delta_beta(cls, beta_true_deg, beta_alg_deg):
        """
        Compute beta_alg_deg - abs(beta_true_deg).

        scalar, scalar: return a scalar
        vector, scalar: return a vector (each vector value compared to scalar)
        vector, vector: vectors should be same size. compare elementwise.
        """

        # type conversion
        beta_true_deg = np.array(beta_true_deg)
        beta_alg_deg = np.array(beta_alg_deg)

        dbeta = beta_alg_deg - np.abs(beta_true_deg)

        return dbeta


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

    class_name = 'UncertaintyParameter'
    data_format = (
        ClassAttr('name', str),
        ClassAttr('fit_name', str),
        ClassAttr('value', float),
        ClassAttr('uncertainty', float, is_sometimes_list=True),
        ClassAttr('units', str),
        ClassAttr('axis_min', float),
        ClassAttr('axis_max', float),
    )

    def __init__(self, name=None, fit_name=None,
                 value=None, uncertainty=None, units=None,
                 axis_min=None, axis_max=None):
        """
        All arguments are required except uncertainty, axis_min, axis_max.
        """

        self.name = name
        self.fit_name = fit_name
        self.value = value
        self.uncertainty = uncertainty
        self.units = units
        self.axis_min = axis_min
        self.axis_max = axis_max

        self.input_check()

    def input_check(self):
        """
        Check data types of inputs. And convert if applicable.
        """

        # error check
        if type(self.name) is not str:
            raise InputError(
                'UncertaintyParameter name should be a string')
        if type(self.fit_name) is not str:
            raise InputError(
                'UncertaintyParameter fit_name should be a string')
        if (not isinstance(self.value, float) and
                not isinstance(self.value, int)):
            raise InputError(
                'UncertaintyParameter value should be a float')

        if type(self.uncertainty) in (list, tuple, np.ndarray):
            if len(self.uncertainty) == 1:
                self.uncertainty = self.uncertainty[0]
            elif len(self.uncertainty) == 2:
                if (not isinstance(self.uncertainty[0], float) and
                        not isinstance(self.uncertainty[0], int) and
                        self.uncertainty[0] is not None):
                    raise InputError(
                        'UncertaintyParameter uncertainty of ' +
                        'length 2 should contain floats or ints')
                if (not isinstance(self.uncertainty[1], float) and
                        not isinstance(self.uncertainty[1], int) and
                        self.uncertainty[1] is not None):
                    raise InputError(
                        'UncertaintyParameter uncertainty of ' +
                        'length 2 should contain floats or ints')
            else:
                raise InputError(
                    'UncertaintyParameter uncertainty should be' +
                    ' of length 1 or 2')
        if (type(self.uncertainty) not in (list, tuple, np.ndarray, type(None))
                and not isinstance(self.uncertainty, float) and
                not isinstance(self.uncertainty, int)):
            raise InputError(
                'UncertaintyParameter uncertainty of ' +
                'length 1 should be a float')

        if type(self.units) is not str:
            raise InputError(
                'UncertaintyParameter units should be a string')
        if (self.axis_min is not None and
                not isinstance(self.axis_min, int) and
                not isinstance(self.axis_min, float)):
            raise InputError(
                'UncertaintyParameter axis_min should be a float')
        if (self.axis_max is not None and
                not isinstance(self.axis_max, int) and
                not isinstance(self.axis_max, float)):
            raise InputError(
                'UncertaintyParameter axis_max should be a float')

        # type conversion
        self.value = float(self.value)
        if self.uncertainty is None:
            pass
        elif isinstance(self.uncertainty, int):
            self.uncertainty = float(self.uncertainty)
        elif type(self.uncertainty) in (list, tuple, np.ndarray):
            self.uncertainty = (float(self.uncertainty[0]),
                                float(self.uncertainty[1]))
        if self.axis_min is not None:
            self.axis_min = float(self.axis_min)
        if self.axis_max is not None:
            self.axis_max = float(self.axis_max)


class AlphaGaussPlusConstant(AlphaUncertainty):
    """
    Fitting d-alpha distribution with gaussian plus constant
    """

    name = 'Alpha Gaussian + constant'
    class_name = 'AlphaGaussPlusConstant'

    def __init__(self, alg_results):
        Uncertainty.__init__(self, alg_results)
        self.append_data_format()

    def append_data_format(self):
        """
        Data format for writing to HDF5 (see trackdata.py)
        """

        data_format = list(self.data_format)
        data_format.extend([
            ClassAttr('nhist', np.ndarray, make_dset=True),
            ClassAttr('xhist', np.ndarray, make_dset=True),
            ClassAttr('resolution', float),
        ])
        # TODO: fit?
        self.data_format = tuple(data_format)

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

        # manual estimate
        halfmax = self.nhist.ptp()/2
        crossing_ind = np.nonzero(self.nhist > halfmax)[0][-1]
        try:
            halfwidth = (
                self.xhist[crossing_ind] +
                (self.xhist[crossing_ind + 1] - self.xhist[crossing_ind]) *
                (halfmax - self.nhist[crossing_ind]) /
                (self.nhist[crossing_ind + 1] - self.nhist[crossing_ind]))
        except IndexError:
            # this occurs when f is small, crossing_ind + 1 may be past
            #   the end of the histogram.
            self.fwhm_estimate = 50
        else:
            self.fwhm_estimate = 2 * halfwidth

    def perform_fit(self):
        """
        """

        # constant + forward peak
        model = lmfit.models.ConstantModel() + lmfit.models.GaussianModel()
        # How this is working:
        #   res and n are defined as parameters, for use in expressions.
        #   f is defined from amplitude.
        #   f_random is constrained to be 1 - f.
        #   c is set not to be varied, but to depend on f_random.
        #   center is fixed to 0.
        #   (FWHM is already defined for the Gaussian.)
        amplitude_estimate = (self.nhist.ptp() * np.sqrt(2*np.pi) *
                              self.fwhm_estimate / 2.355)
        init_values = {'c': self.nhist.min(),
                       'center': 0,
                       'amplitude': amplitude_estimate,
                       'sigma': self.fwhm_estimate / 2.355}
        weights = np.sqrt(1 / (self.nhist + 0.25))  # avoid zeros
        params = model.make_params(**init_values)
        params.add('res', vary=False, value=self.resolution)
        params.add('n', vary=False, value=self.n_values)
        params.add('f', vary=False, expr='amplitude / 2 / res / n')
        params.add('f_random', vary=False, expr='1 - f')
        params['c'].set(value=None, vary=False,
                        expr='res * n * f_random / 180')
        params['center'].vary = False
        self.fit = model.fit(self.nhist, x=self.xhist,
                             params=params, weights=weights)

        if (self.fit.params['f'].value <= 0 or
                self.fit.params['fwhm'].value > 100):
            # bad fit... don't use it
            model = lmfit.models.ConstantModel()
            params = model.make_params(c=self.nhist.mean())
            params.add('res', vary=False, value=self.resolution)
            params.add('n', vary=False, value=self.n_values)
            params.add('f_random', vary=False, value=1.0)
            params.add('fwhm', vary=False, value=np.nan)
            params.add('f', vary=False, value=0.0)
            params.add('center', vary=False, value=0.0)
            self.fit = model.fit(self.nhist, x=self.xhist, params=params)

        if not self.fit.success:
            raise FittingError('Fit failed!')
        else:
            del(self.fwhm_estimate)

    def compute_metrics(self):
        """
        """

        if not self.fit.errorbars:
            fwhm_unc = np.nan
            f_unc = np.nan
            print('FittingWarning: Could not compute errorbars in fit')
        else:
            fwhm_unc = self.fit.params['fwhm'].stderr
            f_unc = self.fit.params['f'].stderr * 100

        fwhm_param = UncertaintyParameter(
            name='FWHM',
            fit_name=self.classname_extract(self),
            value=self.fit.params['fwhm'].value,
            uncertainty=(fwhm_unc, fwhm_unc),
            units='degrees',
            axis_min=0.0,
            axis_max=120.0)
        f_param = UncertaintyParameter(
            name='peak fraction',
            fit_name=self.classname_extract(self),
            value=self.fit.params['f'].value * 100,
            uncertainty=(f_unc, f_unc),
            units='%',
            axis_min=0.0,
            axis_max=100.0)
        self.metrics = {'FWHM': fwhm_param,
                        'f': f_param}


class AlphaGaussPlusConstantPlusBackscatter(AlphaGaussPlusConstant):
    """
    Fitting d-alpha distribution with gaussian + constant + backscatter
    """

    name = 'Alpha Gaussian + backscatter Gaussian + constant'
    class_name = 'AlphaGaussPlusConstantPlusBackscatter'

    # prepare_data is inherited from AlphaGaussPlusConstant

    # perform_fit and compute_metrics have to be defined uniquely

    # data_format remains unchanged

    def perform_fit(self):
        """
        """

        mid = int(round(len(self.nhist)/2))

        # constant + forward peak
        model = (lmfit.models.ConstantModel() +
                 lmfit.models.GaussianModel(prefix='fwd') +
                 lmfit.models.GaussianModel(prefix='bk'))
        fwd_amplitude_estimate = (self.nhist[:mid].ptp() * np.sqrt(2*np.pi) *
                                  self.fwhm_estimate / 2.355)
        bk_amplitude_estimate = (self.nhist[mid:].ptp() * np.sqrt(2*np.pi) *
                                 self.fwhm_estimate * 1.5 / 2.355)
        init_values = {'c': self.nhist.min(),
                       'fwd_center': 0,
                       'fwd_amplitude': fwd_amplitude_estimate,
                       'fwd_sigma': self.fwhm_estimate / 2.355,
                       'bk_center': 180,
                       'bk_amplitude': bk_amplitude_estimate,
                       'bk_sigma': self.fwhm_estimate / 2.355 * 1.5}
        weights = np.sqrt(1 / (self.nhist + 0.25))  # avoid zeros
        params = model.make_params(**init_values)
        params.add('res', vary=False, value=self.resolution)
        params.add('n', vary=False, value=self.n_values)
        params.add('f', vary=False, expr='fwd_amplitude / 2 / res / n')
        params.add('f_bk', vary=False, expr='bk_amplitude / 2 / res / n')
        params.add('f_random', vary=False, expr='1 - f - f_bk')
        params['c'].set(value=None, vary=False,
                        expr='res * n * f_random / 180')
        params['fwd_center'].vary = False
        params['bk_center'].vary = False
        self.fit = model.fit(self.nhist, x=self.xhist,
                             params=params, weights=weights)

        if not self.fit.success:
            raise FittingError('Fit failed!')
        else:
            del(self.fwhm_estimate)

    def compute_metrics(self):
        """
        """

        if not self.fit.errorbars:
            fwhm_unc = np.nan
            f_unc = np.nan
            f_bk_unc = np.nan
            raise FittingWarning('Could not compute errorbars in fit')
        else:
            fwhm_unc = self.fit.params['fwhm'].stderr
            f_unc = self.fit.params['f'].stderr * 100
            f_bk_unc = self.fit.params['f_bk'].stderr * 100

        fwhm_param = UncertaintyParameter(
            name='FWHM',
            fit_name=self.classname_extract(self),
            value=self.fit.params['fwhm'].value,
            uncertainty=(fwhm_unc, fwhm_unc),
            units='degrees',
            axis_min=0.0,
            axis_max=120.0)
        f_param = UncertaintyParameter(
            name='peak fraction',
            fit_name=self.classname_extract(self),
            value=self.fit.params['f'].value * 100,
            uncertainty=(f_unc, f_unc),
            units='%',
            axis_min=0.0,
            axis_max=100.0)
        f_bk_param = UncertaintyParameter(
            name='backscatter fraction',
            fit_name=self.classname_extract(self),
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

    name = 'Alpha 68% containment'
    class_name = 'Alpha68'

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
            fit_name=self.classname_extract(self),
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

    name = 'Beta RMS'
    class_name = 'BetaRms'

    def prepare_data(self):
        pass

    def perform_fit(self):
        self.rms = np.sqrt(np.mean(np.square(self.delta)))

    def compute_metrics(self):
        rms_unc = self.rms / np.sqrt(2*(self.n_values - 1))
        rms_param = UncertaintyParameter(
            name='RMS',
            fit_name=self.classname_extract(self),
            value=self.rms,
            uncertainty=(rms_unc, rms_unc),
            units='degrees',
            axis_min=0.0,
            axis_max=40.0)

        self.metrics = {'RMS': rms_param}


DefaultAlphaUncertainty = AlphaGaussPlusConstant

DefaultBetaUncertainty = BetaRms


##############################################################################
#                               Error classes                                #
##############################################################################


class EvalError(Exception):
    """
    Base class for errors in evaluation.py
    """
    pass


class FittingError(EvalError):
    pass


class InputError(EvalError):
    pass


class SelectionError(EvalError):
    pass


class EvalWarning(Warning):
    """
    Base class for warnings in evaluation.py
    """
    pass


class DataWarning(EvalWarning):
    pass


class FittingWarning(EvalWarning):
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
    assert AlphaUncertainty.delta_alpha(a1, a2) == 10

    # test basic scalar-scalar (floats)
    a1 = 5.5
    a2 = 15.5
    assert AlphaUncertainty.delta_alpha(a1, a2) == 10

    # test wraparound scalar-scalar
    a1 = 175 + 360
    a2 = -175 - 360
    assert AlphaUncertainty.delta_alpha(a1, a2) == 10
    a1 = -175 - 360
    a2 = 175 + 360
    assert AlphaUncertainty.delta_alpha(a1, a2) == -10

    # test vector-scalar (list; ndarray)
    a1 = -175
    a2 = [-150, 30, 175]
    assert np.all(AlphaUncertainty.delta_alpha(a1, a2) ==
                  np.array([25, -155, -10]))
    assert np.all(AlphaUncertainty.delta_alpha(np.array(a1), a2) ==
                  np.array([25, -155, -10]))
    assert np.all(AlphaUncertainty.delta_alpha(a1, np.array(a2)) ==
                  np.array([25, -155, -10]))

    # test vector-vector (list-list; list-ndarray; ndarray-ndarray)
    a1 = [-170, 0, 170]
    a2 = [170.5, 30.5, 150.5]
    assert np.all(AlphaUncertainty.delta_alpha(a1, a2) ==
                  np.array([-19.5, 30.5, -19.5]))
    assert np.all(AlphaUncertainty.delta_alpha(a1, np.array(a2)) ==
                  np.array([-19.5, 30.5, -19.5]))
    assert np.all(AlphaUncertainty.delta_alpha(np.array(a1), np.array(a2)) ==
                  np.array([-19.5, 30.5, -19.5]))

    return None


def test_dbeta():
    """
    """

    # basic scalar-scalar
    b1 = 5
    b2 = 15
    assert BetaUncertainty.delta_beta(b1, b2) == 10

    # absolute value
    b1 = -5
    b2 = 15
    assert BetaUncertainty.delta_beta(b1, b2) == 10

    # floats
    b1 = -5.0
    b2 = 15.0
    assert BetaUncertainty.delta_beta(b1, b2) == 10

    # vector-scalar (list; ndarray)
    b1 = 0
    b2 = [23, 30.0, 0]
    assert np.all(BetaUncertainty.delta_beta(b1, b2) ==
                  np.array([23, 30, 0]))
    assert np.all(BetaUncertainty.delta_beta(np.array(b1), b2) ==
                  np.array([23, 30, 0]))
    assert np.all(BetaUncertainty.delta_beta(b1, np.array(b2)) ==
                  np.array([23, 30, 0]))

    # test vector-vector (list-list; list-ndarray; ndarray-ndarray)
    b1 = [0, -10, 5]
    b2 = [10, 10, 30]
    assert np.all(BetaUncertainty.delta_beta(b1, b2) ==
                  np.array([10, 0, 25]))
    assert np.all(BetaUncertainty.delta_beta(b1, np.array(b2)) ==
                  np.array([10, 0, 25]))
    assert np.all(BetaUncertainty.delta_beta(np.array(b1), np.array(b2)) ==
                  np.array([10, 0, 25]))


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

    dalpha = AlphaUncertainty.delta_alpha(alg_results.alpha_true_deg,
                                          alg_results.alpha_meas_deg)
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
    assert len(AlgorithmResults.data_attrs()) == 8

    # length
    assert len(generate_random_alg_results(length=100)) == 100
    assert len(generate_random_alg_results(has_alpha=False, length=100)) == 100

    # subtests
    test_alg_results_input_check()
    test_alg_results_add()
    test_alg_results_select()
    test_alg_results_from_h5initial()
    test_alg_results_from_track_array()


def test_alg_results_input_check():
    """
    Test AlgorithmResults input check
    """

    # errors
    try:
        AlgorithmResults(filename=5,
                         alpha_true_deg=[10, 20],
                         alpha_meas_deg=[20, 25])
    except InputError:
        pass
    else:
        print('Failed to catch AlgorithmResults filename error')

    try:
        AlgorithmResults(alpha_true_deg=np.random.random(30),
                         alpha_meas_deg=np.random.random(29))
    except InputError:
        pass
    else:
        print('Failed to catch data length mismatch')

    try:
        AlgorithmResults(alpha_true_deg=np.random.random(30))
    except InputError:
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
    except DataWarning:
        pass
    else:
        print('Failed to warn on asymmetric concatenation (None + data)')
    try:
        x = generate_random_alg_results(length=len1, has_beta=True)
        y = generate_random_alg_results(length=len2, has_beta=False)
        z = x + y
        assert z.has_beta is True
        assert np.sum(np.isnan(z.beta_true_deg)) == len2
    except DataWarning:
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
    assert not y.has_beta
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
    except SelectionError:
        pass
    else:
        print('AlgorithmResults.select() failed to raise error with no beta')

    try:
        x.select(asdf_max=500)
    except SelectionError:
        pass
    else:
        print('AlgorithmResults.select() failed to ' +
              'raise error on bad condition')


def test_alg_results_from_h5initial():
    """
    """

    print('test_alg_results_from_h5initial not implemented yet')

    return None


def test_alg_results_from_track_array():
    """
    """

    print('test_alg_results_from_track_array not implemented yet')

    return None


def test_alg_uncertainty(include_plots):
    """
    Test AlgorithmUncertainty class.
    """

    test_uncertainty_parameter_input()
    test_AGPC()
    test_A68()
    test_beta_uncertainties()
    if include_plots:
        print('test_alg_uncertainty plots not implemented yet')

    return None


def test_uncertainty_parameter_input():
    """
    """

    # basic
    UncertaintyParameter(
        name='asdf', fit_name='qwerty', value=23.2, uncertainty=(0.5, 0.6),
        units='degrees', axis_min=0, axis_max=120)
    # single uncertainty
    UncertaintyParameter(
        name='asdf', fit_name='qwerty', value=23.2, uncertainty=0.5,
        units='degrees', axis_min=0, axis_max=120)
    # unspecified uncertainty
    UncertaintyParameter(
        name='asdf', fit_name='qwerty', value=23.2,
        units='degrees', axis_min=0, axis_max=120)
    # unspecified axes limits
    UncertaintyParameter(
        name='asdf', fit_name='qwerty', value=23.2, uncertainty=(0.5, 0.6),
        units='degrees')

    # error checks
    try:
        UncertaintyParameter(
            fit_name='qwerty', value=23.2,
            uncertainty=(0.5, 0.6), units='degrees', axis_min=0, axis_max=120)
    except InputError:
        pass
    else:
        print('UncertaintyParameter failed to catch missing name')
    try:
        UncertaintyParameter(
            name='asdf', value=23.2,
            uncertainty=(0.5, 0.6), units='degrees', axis_min=0, axis_max=120)
    except InputError:
        pass
    else:
        print('UncertaintyParameter failed to catch missing fit_name')
    try:
        UncertaintyParameter(
            name='asdf', fit_name='qwerty',
            uncertainty=(0.5, 0.6), units='degrees', axis_min=0, axis_max=120)
    except InputError:
        pass
    else:
        print('UncertaintyParameter failed to catch missing value')
    try:
        UncertaintyParameter(
            name='asdf', fit_name='qwerty', value=23.2,
            uncertainty=(0.5, 0.6), axis_min=0, axis_max=120)
    except InputError:
        pass
    else:
        print('UncertaintyParameter failed to catch missing units')
    try:
        UncertaintyParameter(
            name=123, fit_name='qwerty', value=23.2,
            uncertainty=(0.5, 0.6), units='degrees', axis_min=0, axis_max=120)
    except InputError:
        pass
    else:
        print('UncertaintyParameter failed to catch bad name')
    try:
        UncertaintyParameter(
            name='asdf', fit_name='qwerty', value='asdf',
            uncertainty=(0.5, 0.6), units='degrees', axis_min=0, axis_max=120)
    except InputError:
        pass
    else:
        print('UncertaintyParameter failed to catch bad value')
    try:
        UncertaintyParameter(
            name='asdf', fit_name='qwerty', value=23.2,
            uncertainty=(0.5, 0.6, 0.7), units='degrees')
    except InputError:
        pass
    else:
        print('UncertaintyParameter failed to catch bad uncertainty')


def AGPC_basic_assertions(aunc):
    """
    Make some basic assertions about an AlphaGaussPlusConstant object.
    """

    assert aunc.fit.params['center'] == 0
    assert aunc.fit.params['f_random'] + aunc.fit.params['f'] == 1
    assert isinstance(aunc.metrics['f'].value, float)
    assert isinstance(aunc.metrics['FWHM'].value, float)
    assert isinstance(aunc.metrics['f'].uncertainty[0], float)
    assert isinstance(aunc.metrics['FWHM'].uncertainty[0], float)
    assert not np.isnan(aunc.metrics['f'].uncertainty[0])
    assert not np.isnan(aunc.metrics['FWHM'].uncertainty[0])


def test_AGPC():
    """
    Test AlphaGaussPlusConstant class.
    """

    # various levels of statistics
    fwhm = 30
    f = 0.7
    ar = generate_random_alg_results(length=10000, a_f=f, a_fwhm=fwhm)
    agpc = AlphaGaussPlusConstant(ar)
    try:
        assert np.abs(agpc.fit.params['fwhm'].value - fwhm)/fwhm < 0.05
        assert np.abs(agpc.fit.params['f'].value - f)/f < 0.05
    except AssertionError:
        print('Bad result while testing AlphaGaussPlusConstant, ' +
              'length = 10000. It is possible this is random chance, ' +
              'please try again.')
    AGPC_basic_assertions(agpc)

    ar = generate_random_alg_results(length=1000000, a_f=f, a_fwhm=fwhm)
    agpc = AlphaGaussPlusConstant(ar)
    try:
        assert np.abs(agpc.fit.params['fwhm'].value - fwhm)/fwhm < 0.02
        assert np.abs(agpc.fit.params['f'].value - f)/f < 0.02
    except AssertionError:
        print('Bad result while testing AlphaGaussPlusConstant, ' +
              'length = 1000000. It is possible this is random chance, ' +
              'please try again.')
    AGPC_basic_assertions(agpc)

    ar = generate_random_alg_results(length=100, a_f=f, a_fwhm=fwhm)
    agpc = AlphaGaussPlusConstant(ar)
    try:
        assert np.abs(agpc.fit.params['fwhm'].value - fwhm)/fwhm < 0.25
        assert np.abs(agpc.fit.params['f'].value - f)/f < 0.25
    except AssertionError:
        print agpc.fit.params['fwhm'].value
        print agpc.fit.params['f'].value
        print('Bad result while testing AlphaGaussPlusConstant, ' +
              'length = 100. It is possible this is random chance, ' +
              'please try again.')
    AGPC_basic_assertions(agpc)

    ar = generate_random_alg_results(length=25, a_f=f, a_fwhm=fwhm)
    agpc = AlphaGaussPlusConstant(ar)
    AGPC_basic_assertions(agpc)

    # edge cases of f
    fwhm = 30
    f = 1.0
    ar = generate_random_alg_results(length=100, a_f=f, a_fwhm=fwhm)
    agpc = AlphaGaussPlusConstant(ar)
    try:
        assert np.abs(agpc.fit.params['fwhm'].value - fwhm)/fwhm < 0.25
        assert np.abs(agpc.fit.params['f'].value - f)/f < 0.25
    except AssertionError:
        print('Bad result while testing AlphaGaussPlusConstant, ' +
              'length = 100 / f = 1.0. It is possible this is random ' +
              'chance, please try again.')
    AGPC_basic_assertions(agpc)

    fwhm = 50
    f = 0.1
    ar = generate_random_alg_results(length=1000, a_f=f, a_fwhm=fwhm)
    agpc = AlphaGaussPlusConstant(ar)
    try:
        assert agpc.fit.params['f'].value < 0.2
    except AssertionError:
        print('Bad result while testing AlphaGaussPlusConstant, ' +
              'length = 1000 / f = 0.1. It is possible this is random ' +
              'chance, please try again.')
    AGPC_basic_assertions(agpc)

    # explore for errors
    for fwhm in [10, 25, 40, 60, 90]:
        for f in [0, 0.1, 0.2, 0.5, 0.8, 1]:
            for L in [50, 100, 1000, 10000]:
                ar = generate_random_alg_results(length=L, a_f=f, a_fwhm=fwhm)
                agpc = AlphaGaussPlusConstant(ar)
                try:
                    AGPC_basic_assertions(agpc)
                except AssertionError:
                    print ('* issue with AGPC at ' +
                           'FWHM = {}, f = {}, length = {}').format(fwhm, f, L)


def A68_basic_assertions(aunc):
    """
    Make some basic assertions about an Alpha68 object.
    """

    assert isinstance(aunc.metrics['contains68'].value, float)
    assert isinstance(aunc.metrics['contains68'].uncertainty[0], float)
    assert not np.isnan(aunc.metrics['contains68'].uncertainty[0])


def test_A68():
    """
    """

    ar = generate_random_alg_results(length=1000, a_f=0.7, a_fwhm=30)
    a68 = Alpha68(ar)
    A68_basic_assertions(a68)

    ar = generate_random_alg_results(length=100, a_f=0.7, a_fwhm=30)
    a68 = Alpha68(ar)
    A68_basic_assertions(a68)

    ar = generate_random_alg_results(length=10000, a_f=0.7, a_fwhm=30)
    a68 = Alpha68(ar)
    A68_basic_assertions(a68)

    ar = generate_random_alg_results(length=1000, a_f=1.0, a_fwhm=30)
    a68 = Alpha68(ar)
    A68_basic_assertions(a68)

    ar = generate_random_alg_results(length=1000, a_f=0.2, a_fwhm=50)
    a68 = Alpha68(ar)
    A68_basic_assertions(a68)

    # explore for errors
    for fwhm in [10, 25, 40, 60, 90]:
        for f in [0, 0.1, 0.2, 0.5, 0.8, 1]:
            for L in [50, 100, 1000, 10000]:
                ar = generate_random_alg_results(length=L, a_f=f, a_fwhm=fwhm)
                a68 = Alpha68(ar)
                try:
                    A68_basic_assertions(a68)
                except AssertionError:
                    print ('* issue with A68 at ' +
                           'FWHM = {}, f = {}, length = {}').format(fwhm, f, L)


def beta_RMS_basic_assertions(bunc):
    """
    Make some basic assertions about a BetaRms object.
    """

    assert isinstance(bunc.metrics['RMS'].value, float)
    assert isinstance(bunc.metrics['RMS'].uncertainty[0], float)
    assert not np.isnan(bunc.metrics['RMS'].uncertainty[0])


def test_beta_uncertainties():
    """
    """
    ar = generate_random_alg_results(length=1000)
    brms = BetaRms(ar)
    beta_RMS_basic_assertions(brms)

    for f in [0, 0.1, 0.2, 0.5, 0.8, 1]:
        for L in [50, 100, 1000, 10000]:
            ar = generate_random_alg_results(length=L, b_f=f)
            brms = BetaRms(ar)
            try:
                beta_RMS_basic_assertions(brms)
            except AssertionError:
                print ('* issue with BetaRms at ' +
                       'b_f = {}, length = {}').format(f, L)


def test_comprehensive():
    """
    Test AlgorithmResults and AlgorithmUncertainty together.
    """

    # TODO
    # include AlgorithmResults.add_uncertainty, etc.
    print('test_comprehensive not implemented yet')


if __name__ == '__main__':
    """
    Run tests.
    """

    include_plots = True

    test_alg_results()
    test_dalpha()
    test_dbeta()
    test_alg_uncertainty(include_plots)
    test_comprehensive()
