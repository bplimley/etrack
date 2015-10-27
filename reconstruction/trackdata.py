#!/usr/bin/python

import numpy as np
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

    def __init__(self, image,
                 is_modeled=None, is_measured=None, is_experimental=None,
                 pixel_size_um=None, noise_ev=None, g4track=None,
                 energy_kev=None, x_offset_pix=None, y_offset_pix=None,
                 timestamp=None, shutter_ind=None, label=None):
        """
        Construct a track object.

        Required input:
          image

        Either is_modeled or is_measured is required.

        Keyword inputs:
          is_modeled (bool)
          is_measured (bool)
          is_experimental (bool)
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

        # handle flags
        if (
                is_modeled is None and
                is_measured is None and
                is_experimental is None):
            raise InputError('Please specify modeled or measured!')
        elif ((is_modeled is True and is_measured is True) or
              (is_modeled is True and is_experimental is True)):
            raise InputError('Track cannot be both modeled and measured!')
        elif is_measured is not None:
            self.is_experimental = self.is_measured = bool(is_measured)
            self.is_modeled = not bool(is_measured)
        elif is_experimental is not None:
            self.is_measured = self.is_experimental = bool(is_experimental)
            self.is_modeled = not bool(is_measured)
        elif is_modeled is not None:
            self.is_modeled = bool(is_modeled)
            self.is_measured = self.is_experimental = not bool(is_modeled)

        if g4track is not None:
            assert type(g4track) is G4Track
            # or handle e.g. a g4 matrix input

        self.image = np.array(image)

        if pixel_size_um is not None:
            pixel_size_um = np.float(pixel_size_um)
        self.pixel_size_um = pixel_size_um

        if noise_ev is not None:
            noise_ev = np.float(noise_ev)
        self.noise_ev = noise_ev

        self.g4track = g4track

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

        if timestamp is not None and type(timestamp) is not datetime.datetime:
                raise InputError('timestamp should be a datetime object')
        self.timestamp = timestamp

        self.label = str(label)

        self.algorithms = {}

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

def write_alg_results_to_hdf5(alg_results, h5group):
    """
    what does it take to write a class to file?
    """

    if alg_results.parent is not None:
        for i, parent in enumerate(alg_results.parent):
            groupname = 'parent'
            if groupname not in h5group:
                parent_group = h5group.create_group(groupname)

            parent_index_str = str(i)
            this_group = parent_group.create_group(parent_index_str)
            write_alg_results_to_hdf5(parent, this_group)

    if alg_results.filename is not None:
        for i, filename in enumerate(alg_results.filename):
            groupname = 'filename'
            if groupname not in h5group:
                filename_group = h5group.create_group(groupname)
            filename_index_str = str(i)
            filename_group.attrs.create(filename_index_str, filename)

    h5group.attrs.create('has_alpha', alg_results.has_alpha, dtype=bool)
    h5group.attrs.create('has_beta', alg_results.has_beta, dtype=bool)
    h5group.attrs.create('data_length', alg_results.data_length, dtype=int)

    L = alg_results.data_length
    h5group.create_dataset(
        'alpha_true_deg', (L,), dtype=float, data=alg_results.alpha_true_deg)
    h5group.create_dataset(
        'beta_true_deg', (L,), dtype=float, data=alg_results.beta_true_deg)

    for i, unc in enumerate(alg_results.uncertainty_list):
        groupname = 'uncertainty_list'
        if groupname not in h5group:
            unc_list_group = h5group.create_group(groupname)

        unc_list_index_str = str(i)
        this_unc_group = unc_list_group.create_group(unc_list_index_str)
        write_unc_to_hdf5(unc, this_unc_group)


def write_object_to_hdf5(obj, h5group):
    """
    Take the user-defined class instance, obj, and write it to HDF5.

    Requires data_format to be an attribute of the object.

    h5group is the location it is written to file. The class attributes are
    attributes in h5group, datasets in h5group, or subgroups of h5group.
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
                'RuntimeError on ' + h5group + '.file - please confirm file ' +
                'is not already closed')
        if h5group.file.mode != 'r+':
            raise InterfaceError(
                'Cannot write object to h5file in read-only mode')

    def attr_check(attr):
        """
        check for disallowed value
        """

        try:
            data = getattr(obj, attr.name)
        except AttributeError:
            raise InterfaceError('Attribute does not exist')

        if attr.may_be_none and data is None:
            raise InterfaceError('Found unexpected "None" value')
        if (attr.is_always_list and
                not isinstance(data, list) and
                not isinstance(data, tuple)):
            raise InterfaceError('Expected a list type')
        if (not attr.is_always_list and not attr.is_sometimes_list and
                (isinstance(data, list) or isinstance(data, tuple))):
            raise InterfaceError('Found unexpected list type')
        if (attr.is_always_dict and not isinstance(data, dict)):
            raise InterfaceError('Expected a dict type')
        if (data is not None and attr.is_user_object and
                (isinstance(data, int) or
                 isinstance(data, float) or
                 isinstance(data, np.ndarray) or
                 isinstance(data, str))):
            raise InterfaceError(
                'Expected a user object, found a basic data type')
        if attr.dtype is not None:
            try:
                attr.dtype(data)
            except ValueError:
                raise InterfaceError('Attribute data cannot be cast properly')

        return data

    def write_one_item(attr, data, h5group):
        """
        Write one hdf5 dataset or hdf5 attribute to the hdf5 file.

        Inputs:
        attr: the ClassAttr object describing this attribute.
        data: the value of the attribute in this instance.
        h5group: the hdf5 file or group object in which the object should be
            written.
        """

        if attr.is_user_object:
            this_obj_group = h5group.create_group(attr.name)
            # recurse
            # TODO: detect multiple references to the same object, and link
            write_object_to_hdf5(data, this_obj_group)
        elif attr.make_dset:
            h5group.create_dataset(
                attr.name, shape=np.shape(data), dtype=attr.dtype, data=data)
        else:
            h5group.attrs.create(
                attr.name, data, shape=np.shape(data), dtype=attr.dtype)

    # ~~~ begin main ~~~
    input_check(obj, h5group)

    for attr in obj.data_format:
        data = attr_check(attr)

        # if None: skip
        if attr.may_be_none and data is None:
            continue

        is_list = isinstance(data, list) or isinstance(data, tuple)
        if is_list:
            subgroup = h5group.create_group(attr.name)
            for i, el in enumerate(data):
                write_one_item(attr, el, subgroup)
        else:
            write_one_item(attr, data, h5group)


class ClassAttr(object):
    """
    Description of one attribute of a class, for the purposes of saving to file
    and loading from file.
    """

    def __init__(self, name, dtype,
                 make_dset=False,
                 may_be_none=False,
                 is_always_list=False,
                 is_sometimes_list=False,
                 is_always_dict=False,
                 is_user_object=False):
        self.name = name
        self.dtype = dtype
        self.make_dset = make_dset
        self.may_be_none = may_be_none
        self.is_always_list = is_always_list
        self.is_sometimes_list = is_sometimes_list
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
    except RuntimeError:
        pass

    try:
        Track(image, is_modeled=True, is_measured=True)
        raise RuntimeError('Failed to raise error on Track instantiation')
    except RuntimeError:
        pass

    try:
        Track(image, is_modeled=False, is_measured=False)
        raise RuntimeError('Failed to raise error on Track instantiation')
    except RuntimeError:
        pass

    try:
        Track(image, is_experimental=True, is_measured=False)
        raise RuntimeError('Failed to raise error on Track instantiation')
    except RuntimeError:
        pass


def test_G4Track():
    """
    """

    G4Track(matrix=None, alpha_deg=132.5, beta_deg=-43.5, energy_tot_kev=201.2)

    # G4Track(matrix=test_matrix())


if __name__ == '__main__':
    """
    Run tests.
    """

    import h5py

    test_G4Track()
    test_Track()
    test_TrackExceptions()
    test_AlgorithmOutput()

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
