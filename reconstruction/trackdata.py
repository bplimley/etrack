#!/usr/bin/python

import numpy as np
import h5py
import datetime
import ipdb as pdb

import hybridtrack
import evaluation


##############################################################################
#                                  G4Track                                   #
##############################################################################


class G4Track(object):
    """
    Electron track from Geant4.
    """

    __version__ = '0.1'

    def __init__(self,
                 matrix=None,
                 alpha_deg=None, beta_deg=None,
                 energy_tot_kev=None, energy_dep_kev=None, energy_esc_kev=None,
                 x=None, dE=None, depth_um=None,
                 is_contained=None):
        """
        Construct G4Track object.

        If matrix is supplied and other quantities are not, then the other
        quantities will be calculated using the matrix (not implemented yet).

          matrix
          alpha_deg
          beta_deg
          energy_tot_kev
          energy_dep_kev
          energy_esc_kev
          x
          dE
          depth_um
          is_contained
        """

        self.matrix = matrix

        if matrix is not None and (
                x is None or dE is None or
                energy_tot_kev is None or energy_dep_kev is None or
                energy_esc_kev is None or x is None or dE is None or
                depth_um is None or is_contained is None):
            self.measure_quantities()
        else:
            self.alpha_deg = alpha_deg
            self.beta_deg = beta_deg
            self.energy_tot_kev = energy_tot_kev
            self.energy_dep_kev = energy_dep_kev
            self.energy_esc_kev = energy_esc_kev
            self.x = x
            self.dE = dE
            self.depth_um = depth_um
            self.is_contained = is_contained

    @classmethod
    def from_h5initial(cls, evt):
        """
        Construct a G4Track instance from an event in an HDF5 file.

        The format of the HDF5 file is 'initial', a.k.a. the mess that Brian
        first made in September 2015.
        """

        alpha = evt.attrs['cheat_alpha']
        beta = evt.attrs['cheat_beta']
        energy_tot = evt.attrs['Etot']
        energy_dep = evt.attrs['Edep']
        track = cls(
            matrix=None,
            alpha_deg=alpha, beta_deg=beta,
            energy_tot_kev=energy_tot, energy_dep_kev=energy_dep,
            energy_esc_kev=None,
            x=None, dE=None, depth_um=None,
            is_contained=None)

        return track

    def measure_quantities(self):
        """
        Measure the following using the Geant4 matrix.

          alpha
          beta
          energy_tot
          energy_dep
          energy_esc
          x
          dE
          depth_um
          is_contained
        """

        # TODO
        raise NotImplementedError("haven't written this yet")

        if self.matrix is None:
            raise DataError('measure_quantities needs a geant4 matrix')

    # TODO:
    # add interface for referencing Track objects associated with this G4Track
    #
    # dictionary-style or ...?
    #
    # see http://www.diveintopython3.net/special-method-names.html


##############################################################################
#                                    Track                                   #
##############################################################################


class Track(object):
    """
    Electron track, from modeling or from experiment.
    """

    __version__ = '0.1'

    def __init__(self, image, **kwargs):
        """
        Construct a track object.

        Required input:
          image

        Either is_modeled or is_measured is required.

        Keyword inputs:
          is_modeled (bool)
          is_measured (bool)
          pixel_size_um (float)
          noise_ev (float)
          g4track (G4Track object)
          energy_kev (float)
          x_offset_pix (int)
          y_offset_pix (int)
          timestamp (datetime object)
          shutter_ind (int)
          label (string)
        """

        self.get_data_format()

        self.input_handling(image, **kwargs)

        self.algorithms = {}

    def get_data_format(self):
        """
        Data format for writing to HDF5 (see trackdata.py)
        """

        self.data_format = (
            ClassAttr('is_modeled', bool),
            ClassAttr('pixel_size_um', float),
            ClassAttr('noise_ev', float, may_be_none=True),
            ClassAttr('g4track', G4Track,
                      may_be_none=True, is_user_object=True),
            ClassAttr('energy_kev', float),
            ClassAttr('x_offset_pix', int, may_be_none=True),
            ClassAttr('y_offset_pix', int, may_be_none=True),
            ClassAttr('timestamp', str, may_be_none=True),
            ClassAttr('shutter_ind', int, may_be_none=True),
            ClassAttr('label', str, may_be_none=True),
        )

    def input_handling(self, image,
                       is_modeled=None, is_measured=None,
                       pixel_size_um=None, noise_ev=None,
                       g4track=None,
                       energy_kev=None,
                       x_offset_pix=None, y_offset_pix=None,
                       timestamp=None, shutter_ind=None,
                       label=None):

        if is_modeled is None and is_measured is None:
            raise InputError('Please specify modeled or measured!')
        elif is_modeled is True and is_measured is True:
            raise InputError('Track cannot be both modeled and measured!')
        elif is_modeled is False and is_measured is False:
            raise InputError('Track must be either modeled or measured!')
        elif is_measured is not None:
            self.is_measured = bool(is_measured)
            self.is_modeled = not bool(is_measured)
        elif is_modeled is not None:
            self.is_modeled = bool(is_modeled)
            self.is_measured = not bool(is_modeled)

        if g4track is not None and not isinstance(g4track, G4Track):
            raise InputError('g4track input must be a G4Track!')
            # or handle e.g. a g4 matrix input
        self.g4track = g4track

        self.image = np.array(image)

        if pixel_size_um is not None:
            pixel_size_um = np.float(pixel_size_um)
        self.pixel_size_um = pixel_size_um

        if noise_ev is not None:
            noise_ev = np.float(noise_ev)
        self.noise_ev = noise_ev

        if energy_kev is not None:
            energy_kev = np.float(energy_kev)
        self.energy_kev = energy_kev

        if x_offset_pix is not None:
            np.testing.assert_almost_equal(
                x_offset_pix, np.round(x_offset_pix), decimal=3,
                err_msg='x_offset_pix must be an integer')
            x_offset_pix = int(np.round(x_offset_pix))
        self.x_offset_pix = x_offset_pix

        if y_offset_pix is not None:
            np.testing.assert_almost_equal(
                y_offset_pix, np.round(y_offset_pix), decimal=3,
                err_msg='y_offset_pix must be an integer')
            y_offset_pix = int(np.round(y_offset_pix))
        self.y_offset_pix = y_offset_pix

        if shutter_ind is not None:
            np.testing.assert_almost_equal(
                shutter_ind, np.round(shutter_ind), decimal=3,
                err_msg='shutter_ind must be an integer')
            shutter_ind = int(np.round(shutter_ind))
        self.shutter_ind = shutter_ind

        if (timestamp is not None and
                not isinstance(timestamp, datetime.datetime)):
            raise InputError('timestamp should be a datetime object')
        self.timestamp = str(timestamp)

        if label is not None:
            label = str(label)
        self.label = label

    @classmethod
    def from_h5initial_all(cls, evt):
        """
        Construct a dictionary of Track objects from an event in an HDF5 file.

        type(output['pix10_5noise0']) = Track
        """

        tracks = {}
        g4track = G4Track.from_h5initial(evt)
        for fieldname in evt:
            if fieldname.startswith('pix'):
                tracks[fieldname] = cls.from_h5initial_one(evt[fieldname],
                                                           g4track=g4track)

    @classmethod
    def from_h5initial_one(cls, diffusedtrack, g4track=None):
        """
        Construct a Track object from one pixelsize/noise of an event in an
        HDF5 file.
        """

        image = diffusedtrack['img']
        pix = diffusedtrack.attrs['pixel_size_um']
        noise = diffusedtrack.attrs['noise_ev']

        track = Track(image,
                      is_modeled=True, pixel_size_um=pix, noise_ev=noise,
                      g4track=g4track, label='MultiAngle h5 initial')
        if 'matlab_alpha' in diffusedtrack.attrs:
            alpha = diffusedtrack.attrs['matlab_alpha']
            track.add_algorithm('matlab HT v1.5',
                                alpha_deg=alpha, beta_deg=None)

        return track

    @classmethod
    def load(cls, filename):
        """
        Load a track from file (saved using track.save).
        """

        raise NotImplementedError('Loading not implemented yet!')

    def add_algorithm(self, alg_name, alpha_deg, beta_deg, info=None):
        """
        """

        if alg_name in self.algorithms:
            raise InputError(alg_name + " already in algorithms")
        self.algorithms[alg_name] = AlgorithmOutput(
            alg_name, alpha_deg, beta_deg, info=info)

    def list_algorithms(self):
        """
        List AlgorithmOutput objects attached to this track.

        Like dict.keys()
        """

        return self.algorithms.keys()

    def keys(self):
        """
        Allow dictionary-like behavior on Track object for its attached
        algorithms.
        """

        return self.list_algorithms()

    def __getitem__(self, key):
        # Map dictionary lookup to algorithms dictionary.
        return self.algorithms[key]

    def __contains__(self, item):
        # Map dictionary lookup to algorithms dictionary.
        return item in self.algorithms

    def save(self, filename):
        """
        Write all track data to an HDF5 file.
        """

        raise NotImplementedError('Saving not implemented yet!')


##############################################################################
#                            Algorithm Output                                #
##############################################################################


class AlgorithmOutput(object):
    """
    The result of one specific reconstruction algorithm, on one track.

    __init__ inputs:
      alg_name: string representing what algorithm this is from
        (e.g., "matlab HT v1.5")
      alpha_deg: the alpha value measured by this algorithm, in degrees
      beta_deg: the beta value measured by this algorithm, in degrees
      info (optional): can contain more information from algorithm
    """

    __version__ = '0.1'

    def __init__(self, alg_name, alpha_deg, beta_deg, info=None):
        self.alpha_deg = alpha_deg
        self.beta_deg = beta_deg
        self.alg_name = alg_name
        self.info = info


##############################################################################
#                               Error classes                                #
##############################################################################

class TrackDataError(Exception):
    """
    Base class for errors in trackdata.py
    """
    pass


class DataError(TrackDataError):
    pass


class InputError(TrackDataError):
    pass


class InterfaceError(TrackDataError):
    pass


##############################################################################
#                                    I/O                                     #
##############################################################################

def write_object_to_hdf5(obj, h5group):
    """
    Take the user-defined class instance, obj, and write it to HDF5.

    Requires data_format to be an attribute of the object.

    h5group is an existing h5py file/group to write to. The class attributes
    are attributes in h5group, datasets in h5group, or subgroups of h5group.
    """

    def input_check(obj, h5group):
        """
        """

        if not hasattr(obj, 'data_format'):
            raise InterfaceError(
                'Need attribute data_format in order to write object to HDF5')
        if not isinstance(h5group, h5py.Group):
            raise InterfaceError(
                'h5group should be a file or group from h5py')
        try:
            if not isinstance(h5group.file, h5py.File):
                raise InterfaceError(
                    'h5group should be a file or group from h5py')
        except RuntimeError:
            raise InterfaceError(
                'RuntimeError on file attribute - please confirm file ' +
                'is not already closed')
        if h5group.file.mode != 'r+':
            raise InterfaceError(
                'Cannot write object to h5file in read-only mode')

    def attr_check(obj, attr):
        """
        attribute may be None, list/tuple, dict, or "singular" object
        """

        # get attribute
        try:
            data = getattr(obj, attr.name)
        except AttributeError:
            raise InterfaceError('Attribute does not exist')

        # check for disallowed None
        if not attr.may_be_none and data is None:
            raise InterfaceError('Found unexpected "None" value')

        if data is None:
            return data
            # remaining checks are not applicable

        if isinstance(data, list) or isinstance(data, tuple):
            # check if list/tuple is disallowed
            if not attr.is_always_list and not attr.is_sometimes_list:
                raise InterfaceError('Found unexpected list type')
            # item checks occur later, in main
        elif isinstance(data, dict):
            # check if dict is disallowed
            if not attr.is_always_dict and not attr.is_sometimes_dict:
                raise InterfaceError('Found unexpected dict type')
            # item checks occur later, in main
        else:
            if attr.is_always_list:
                raise InterfaceError('Expected a list type')
            if attr.is_always_dict:
                raise InterfaceError('Expected a dict type')
            item_check(attr, data)

        return data

    def item_check(attr, item):
        """
        item must not be a list/tuple or dict. it is a "singular" object.

        Therefore, ignore the list and dict property flags of attr.
        """

        # None checks
        if not attr.may_be_none and item is None:
            raise InterfaceError('Found unexpected "None" value')
        if item is None:
            return item
            # remaining checks are not applicable, and no need to return item

        # other types. exact type not required, but must be castable
        if attr.is_user_object:
            # must be strict about type
            if not isinstance(item, attr.dtype):
                raise InterfaceError(
                    'Expected user object of type ' + str(attr.dtype) +
                    ', found a ' + str(type(item)))
            # other checks performed in a sub-call of write_object_to_hdf5
        elif attr.dtype is np.ndarray:
            try:
                item = np.array(item)
            except ValueError:
                raise InterfaceError('Attribute data cannot be cast properly')
        else:
            try:
                item = attr.dtype(item)
            except ValueError:
                raise InterfaceError('Attribute data cannot be cast properly')

        return item

    def write_item(attr, name, data, h5group):
        """
        Write one item to the hdf5 file.

        Inputs:
          attr: the ClassAttr object describing this attribute.
          name: name for the new hdf5 object
                (either attr.name or the dict key)
          data: data to put in the object (intelligently)
                (either attr.data or the dict value)
          h5group: parent location of the new hdf5 object
        """

        if attr.is_user_object:
            this_obj_group = h5group.create_group(name)
            this_obj_group.attrs.create('obj_type', data=data.class_name)
            # recurse
            # TODO: detect multiple references to the same object, and link
            write_object_to_hdf5(data, this_obj_group)
        elif attr.make_dset:
            h5group.create_dataset(
                name, shape=np.shape(data), data=data)
        else:
            h5group.attrs.create(
                name, data, shape=np.shape(data))

    # ~~~ begin main ~~~
    input_check(obj, h5group)

    for attr in obj.data_format:
        data = attr_check(obj, attr)

        # if None: skip
        if data is None:
            continue

        is_list = isinstance(data, list) or isinstance(data, tuple)
        is_dict = isinstance(data, dict)
        if is_list:
            subgroup = h5group.create_group(attr.name)
            subgroup.attrs.create('obj_type', data='list')
            for i, item in enumerate(data):
                item = item_check(attr, item)
                write_item(attr, str(i), item, subgroup)
        elif is_dict:
            subgroup = h5group.create_group(attr.name)
            subgroup.attrs.create('obj_type', data='dict')
            for key, item in data.items():
                item = item_check(attr, item)
                write_item(attr, key, item, subgroup)
        else:
            write_item(attr, attr.name, data, h5group)


def read_object_from_hdf5(h5group):
    """
    Take an HDF5 group which represents a class instance, parse and return it.

    The class definition should exist in trackdata.py, evaluation.py.
    """

    def input_check(h5group):
        """
        Input HDF5 group should have 'obj_type' attribute which matches
        a class we know.
        """

        if 'obj_type' not in h5group.attrs:
            raise InterfaceError(
                'HDF5 object should have an attribute, obj_type')
        obj_type = h5group.attrs['obj_type']

        if obj_type == 'AlgorithmResults':
            data_format = evaluation.AlgorithmResults

    # ~~~ begin main ~~~
    input_check(h5group)




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


##############################################################################
#                                  Testing                                   #
##############################################################################


def test_AlgorithmOutput():
    """
    """

    AlgorithmOutput('matlab HT v1.5', 120.5, 43.5)


def test_Track():
    """
    """

    image = hybridtrack.test_input()

    track = Track(image, is_modeled=True, pixel_size_um=10.5, noise_ev=0.0,
                  label='MultiAngle', energy_kev=np.sum(image))
    options, info = hybridtrack.reconstruct(image)

    track.add_algorithm('matlab HT v1.5', 120.5, 43.5, info=info)


def test_TrackExceptions():
    """
    """

    image = hybridtrack.test_input()
    try:
        Track(image)
        raise RuntimeError('Failed to raise error on Track instantiation')
    except InputError:
        pass

    try:
        Track(image, is_modeled=True, is_measured=True)
        raise RuntimeError('Failed to raise error on Track instantiation')
    except InputError:
        pass

    try:
        Track(image, is_modeled=False, is_measured=False)
        raise RuntimeError('Failed to raise error on Track instantiation')
    except InputError:
        pass


def test_G4Track():
    """
    """

    G4Track(matrix=None, alpha_deg=132.5, beta_deg=-43.5, energy_tot_kev=201.2)

    # G4Track(matrix=test_matrix())


class TestIO(object):
    """
    For testing data_format handling in IO.
    """

    def __init__(self, data_format, **kwargs):
        """
        Initialize a TestIO object with a user-defined data_format and
        list of keyword arguments.
        """

        self.data_format = data_format
        for key, val in kwargs.iteritems():
            setattr(self, key, val)


def test_write():
    """
    """

    import os

    filebase = ''.join(chr(i) for i in np.random.randint(97, 122, size=(8,)))
    filename = '.'.join([filebase, 'h5'])

    # TestIO objects and ClassAttr
    # test all 'singular' data types
    data_format = (ClassAttr('int1', int), ClassAttr('int2', int),
                   ClassAttr('str1', str), ClassAttr('float1', float),
                   ClassAttr('bool1', bool), ClassAttr('array1', np.ndarray))
    t = TestIO(data_format, int1=34, int2=-6, str1='asdf', float1=3.141,
               bool1=False, array1=np.array([1.0, 1.1, 1.2]))
    with h5py.File(filename) as h5file:
        write_object_to_hdf5(t, h5file)
    os.remove(filename)

    # test list (is_always_list)
    data_format = (ClassAttr('float1', float),
                   ClassAttr('list1', int, is_always_list=True),
                   ClassAttr('str1', str))
    t = TestIO(data_format, float1=-26.3, str1='foo', list1=[1, 3, 5, 7, 9])
    with h5py.File(filename) as h5file:
        write_object_to_hdf5(t, h5file)
    os.remove(filename)

    # test tuple (is_sometimes_list)
    data_format = (ClassAttr('float1', float),
                   ClassAttr('list1', float, is_sometimes_list=True),
                   ClassAttr('str1', str))
    t = TestIO(data_format, float1=-26.3, str1='foo', list1=(1.0, 3.3, 5.1))
    with h5py.File(filename) as h5file:
        write_object_to_hdf5(t, h5file)
    os.remove(filename)

    # test dict (is_always_dict)
    data_format = (ClassAttr('float1', float),
                   ClassAttr('dict1', str, is_always_dict=True),
                   ClassAttr('str1', str))
    t = TestIO(data_format, float1=-26.3, str1='foo',
               dict1={'foo': 'foovalue', 'bar': 'barvalue', 'asdf': 'qwerty'})
    with h5py.File(filename) as h5file:
        write_object_to_hdf5(t, h5file)
    os.remove(filename)

    # test make_dset
    data_format = (ClassAttr('float1', float),
                   ClassAttr('array1', np.ndarray, make_dset=True),
                   ClassAttr('str1', str))
    t = TestIO(data_format, float1=-26.3, str1='foo',
               array1=np.array(range(150)))
    with h5py.File(filename) as h5file:
        write_object_to_hdf5(t, h5file)
    os.remove(filename)

    # test may_be_none
    data_format = (ClassAttr('float1', float),
                   ClassAttr('str1', str, may_be_none=True))
    t = TestIO(data_format, float1=-26.3, str1=None)
    with h5py.File(filename) as h5file:
        write_object_to_hdf5(t, h5file)
    os.remove(filename)

    # test is_sometimes_list without a list
    data_format = (ClassAttr('float1', float),
                   ClassAttr('maybelist1', int, is_sometimes_list=True),
                   ClassAttr('str1', str))
    t = TestIO(data_format, float1=-26.3, str1='foo', maybelist1=3)
    with h5py.File(filename) as h5file:
        write_object_to_hdf5(t, h5file)
    os.remove(filename)

    # test is_sometimes_dict without a dict
    data_format = (ClassAttr('float1', float),
                   ClassAttr('maybedict1', float, is_sometimes_dict=True),
                   ClassAttr('str1', str))
    t = TestIO(data_format, float1=-26.3, str1='foo', maybedict1=3.5)
    with h5py.File(filename) as h5file:
        write_object_to_hdf5(t, h5file)
    os.remove(filename)

    # Real Classes:
    # single user-defined object
    alg_results = evaluation.generate_random_alg_results(length=10000)
    with h5py.File(filename) as h5file:
        write_object_to_hdf5(alg_results, h5file)
    os.remove(filename)

    # multi-level object
    alg_results.add_default_uncertainties()
    with h5py.File(filename) as h5file:
        write_object_to_hdf5(alg_results, h5file)
    os.remove(filename)


def test_read():
    """
    """

    print 'IO read not implemented yet'

if __name__ == '__main__':
    """
    Run tests.
    """

    test_G4Track()
    test_Track()
    test_TrackExceptions()
    test_AlgorithmOutput()
    test_write()
    test_read()

    try:
        h5initial = h5py.File('MultiAngle_HT_11_12.h5', 'r')
    except IOError:
        print('Skipping file tests')
        quit()

    print('Beginning file tests')
    fieldname = 'pix10_5noise0'
    testflag = True
    for i, evt in enumerate(h5initial):
        try:
            g4track = G4Track.from_h5initial(h5initial[evt])
            if fieldname in h5initial[evt]:
                track = Track.from_h5initial_one(h5initial[evt][fieldname])
                track.add_algorithm('test asdf', 122.1, -33.3)
                assert 'test asdf' in track.list_algorithms()
            Track.from_h5initial_all(h5initial[evt])
        except Exception:
            print i
            raise
        if testflag:
            # check g4track
            np.testing.assert_almost_equal(
                g4track.alpha_deg, 61.10767, decimal=4)
            np.testing.assert_almost_equal(
                g4track.beta_deg, 66.98443, decimal=4)
            np.testing.assert_almost_equal(
                g4track.energy_tot_kev, 418.70575, decimal=4)
            np.testing.assert_almost_equal(
                g4track.energy_dep_kev, 418.70575, decimal=4)

            # check track
            np.testing.assert_almost_equal(
                track.noise_ev, 0.0, decimal=4)
            np.testing.assert_almost_equal(
                track.pixel_size_um, 10.5, decimal=4)
            assert track.is_modeled is True
            assert track.is_measured is False

            # only run on first track (that's what these numbers are from)
            testflag = False
