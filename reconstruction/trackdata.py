#!/usr/bin/python

import numpy as np
import h5py
import datetime
import ipdb as pdb
import os

import hybridtrack
import dataformats
from dataformats import ClassAttr


##############################################################################
#                                  G4Track                                   #
##############################################################################


class G4Track(object):
    """
    Electron track from Geant4.
    """

    __version__ = '0.1'
    class_name = 'G4Track'
    # data_format = dataformats.get_format(class_name)

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
    class_name = 'Track'
    data_format = dataformats.get_format(class_name)

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

        self.input_handling(image, **kwargs)

        self.algorithms = {}

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

def write_object_to_hdf5(obj, h5group, name, pyobj_to_h5={}):
    """
    Take the user-defined class instance, obj, and write it to HDF5
    in HDF5 group h5group with name name.

    Requires data_format to be an attribute of the object.

    h5group is an existing h5py file/group to write to. The class attributes
    are attributes in h5group, datasets in h5group, or subgroups of h5group.

    pyobj_to_h5 = object dictionary:
                    {pyobjectA: h5objectA, pyobjectB: h5objectB, ...}
    """

    def check_input(obj, h5group):
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
        # clear pyobj_to_h5 of any closed HDF5 objects
        pyobj_to_h5_copy = pyobj_to_h5.copy()
        for key, val in pyobj_to_h5_copy.iteritems():
            if str(val) == '<Closed HDF5 group>':
                del pyobj_to_h5[key]

    def check_attr(obj, attr):
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
            check_item(attr, data)

        return data

    def check_item(attr, item):
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
            pass
            # can't check type because user-defined dtype has to be just a
            #   classname string

            # # must be strict about type
            # if not isinstance(item, attr.dtype):
            #     raise InterfaceError(
            #         'Expected user object of type ' + str(attr.dtype) +
            #         ', found a ' + str(type(item)))
            # # other checks performed in a sub-call of write_object_to_hdf5
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

    def write_item(attr, name, data, h5group, pyobj_to_h5):
        """
        Write one item to the hdf5 file.

        Inputs:
          attr: the ClassAttr object describing this attribute.
          name: name for the new hdf5 object
                (either attr.name or the dict key)
          data: data to put in the object (intelligently)
                (either attr.data or the dict value)
          h5group: parent location of the new hdf5 object
          pyobj_to_h5: object dictionary to update
        """

        if attr.is_user_object:
            # check id
            if data in pyobj_to_h5:
                # don't write the actual data; make a hard link
                h5group[name] = pyobj_to_h5[data]
            else:
                # recurse
                write_object_to_hdf5(data, h5group, name,
                                     pyobj_to_h5=pyobj_to_h5)
        elif attr.make_dset:
            h5group.create_dataset(
                name, shape=np.shape(data), data=data)
        else:
            h5group.attrs.create(
                name, data, shape=np.shape(data))

    # ~~~ begin main ~~~
    check_input(obj, h5group)

    if obj in pyobj_to_h5:
        h5group[name] = pyobj_to_h5[obj]
        return None
    else:
        if name in h5group:
            del(h5group[name])
        this_group = h5group.create_group(name)
        this_group.attrs.create('obj_type', data=obj.class_name)
        pyobj_to_h5[obj] = this_group

    for attr in obj.data_format:
        data = check_attr(obj, attr)

        # if None: skip
        if data is None:
            continue

        is_list = isinstance(data, list) or isinstance(data, tuple)
        is_dict = isinstance(data, dict)
        if is_list:
            subgroup = this_group.create_group(attr.name)
            subgroup.attrs.create('obj_type', data='list')
            for i, item in enumerate(data):
                item = check_item(attr, item)
                write_item(attr, str(i), item, subgroup, pyobj_to_h5)
        elif is_dict:
            subgroup = this_group.create_group(attr.name)
            subgroup.attrs.create('obj_type', data='dict')
            for key, item in data.items():
                item = check_item(attr, item)
                write_item(attr, key, item, subgroup, pyobj_to_h5)
        else:
            write_item(attr, attr.name, data, this_group, pyobj_to_h5)

    return None


def write_objects_to_hdf5(h5group_or_filename, pyobj_to_h5={}, **kwargs):
    """
    Write a list of objects to file.
    h5group_or_filename, as implied, can be either a h5file/h5group object,
      or a filename for an h5file to create.
      If the filename is missing the .h5 or .hdf5 extension, .h5 will be added.
      And the h5file will be closed if filename is supplied.

    kwargs in the form:
    h5_object_name=py_object
    """

    if (isinstance(h5group_or_filename, str) or
            isinstance(h5group_or_filename, unicode)):
        # filename supplied: create h5 file and close when finished
        filename = h5group_or_filename
        # check extension
        if filename[-5:] != '.hdf5' and filename[-3:] != '.h5':
            filename += '.h5'
        with h5py.File(filename) as h5group:
            for key, val in kwargs.iteritems():
                write_object_to_hdf5(val, h5group, key,
                                     pyobj_to_h5=pyobj_to_h5)
        return filename

    elif isinstance(h5group_or_filename, h5py.Group):
        # h5group supplied: just write the objects
        h5group = h5group_or_filename
        for key, val in kwargs.iteritems():
            write_object_to_hdf5(val, h5group, key,
                                 pyobj_to_h5=pyobj_to_h5)
        return h5group.file.filename

    else:
        raise InterfaceError(
            'write_objects_to_hdf5 needs either an h5py.Group ' +
            'or a string filename')
        return None


def read_object_from_hdf5(h5group, h5_to_pydict={}, ext_data_format=None,
                          verbosity=0):
    """
    Take an HDF5 group which represents a class instance, parse and return it
      as a dictionary of attribute values.

    The class definition should exist in dataformats.py.

    h5_to_pydict = {h5objectA: pyobjectA, h5objectB: pyobjectB, ...}
    """

    def check_input(h5group, data_format):
        """
        Input HDF5 group should have 'obj_type' attribute which matches
        a class we know.
        """

        if ext_data_format is not None:
            # for testing: don't use dataformats.get_format()
            return ext_data_format

        else:
            if 'obj_type' not in h5group.attrs:
                if h5group == h5group.file:
                    raise InterfaceError(
                        'Looks like you supplied the HDF5 file object ' +
                        'instead of the HDF5 group representing the object...')
                else:
                    raise InterfaceError(
                        'HDF5 object should have an attribute, obj_type')
            obj_type = h5group.attrs['obj_type']
            data_format = dataformats.get_format(obj_type)
            return data_format

    def check_attr(h5group, attr):
        """
        Check that the data and attributes in h5group are compatible with the
        data description in attr.

        Return the attribute type: 'none', 'list', 'dict', or 'single'
        Type 'single' includes basic data types and user objects.
        """

        if attr.name in h5group:
            # it is either an h5group or a dataset.
            if 'obj_type' in h5group[attr.name].attrs:
                # if it has the obj_type attribute,
                # it is either a list or a dict or a user-defined object.
                obj_type = h5group[attr.name].attrs['obj_type']
                if obj_type == 'list':
                    if not attr.is_always_list and not attr.is_sometimes_list:
                        raise InterfaceError(
                            'Unexpected list in HDF5 file for attribute ' +
                            '{}'.format(attr.name))
                    hdf5_type = 'list'
                elif obj_type == 'dict':
                    if not attr.is_always_dict and not attr.is_sometimes_dict:
                        raise InterfaceError(
                            'Unexpected dict in HDF5 file for attribute ' +
                            '{}'.format(attr.name))
                    hdf5_type = 'dict'
                else:
                    # user object
                    if attr.is_always_list:
                        raise InterfaceError(
                            'Expected a list in HDF5 file for attribute ' +
                            '{}'.format(attr.name))
                    elif attr.is_always_dict:
                        raise InterfaceError(
                            'Expected a dict in HDF5 file for attribute ' +
                            '{}'.format(attr.name))
                    elif not attr.is_user_object:
                        raise InterfaceError(
                            'Unexpected user object in HDF5 file for ' +
                            'attribute {}'.format(attr.name))
                    hdf5_type = 'single'
            else:
                # not marked with obj_type attribute. A dataset.
                if not attr.make_dset:
                    raise InterfaceError(
                        'Unexpected dataset in HDF5 file for attribute ' +
                        '{}'.format(attr.name))
                hdf5_type = 'single'

        elif attr.name in h5group.attrs:
            # it is an h5 attribute.
            if attr.is_always_list:
                raise InterfaceError(
                    'Expected a list in HDF5 file for attribute ' +
                    '{}'.format(attr.name))
            elif attr.is_always_dict:
                raise InterfaceError(
                    'Expected a dict in HDF5 file for attribute ' +
                    '{}'.format(attr.name))
            elif attr.make_dset:
                raise InterfaceError(
                    'Expected a dataset in HDF5 file for attribute ' +
                    '{}; found HDF5 attribute instead'.format(attr.name))
            elif attr.is_user_object:
                raise InterfaceError(
                    'Expected a user object in HDF5 file for attribute ' +
                    '{}; found HDF5 attribute instead'.format(attr.name))
            hdf5_type = 'single'

        else:
            # not a h5 group, dataset, or attribute. It isn't there.
            if not attr.may_be_none:
                raise InterfaceError(
                    'Failed to find required attribute ' +
                    '{} in HDF5 file'.format(attr.name))
            hdf5_type = 'none'

        return hdf5_type

    def read_item(attr, h5item, h5_to_pydict={}):

        vprint('     Reading item {}'.format(h5item))
        if (not isinstance(h5item, np.ndarray)) and h5item in h5_to_pydict:
            vprint('     Item {} in h5_to_pydict! Skipping'.format(h5item))
            return h5_to_pydict[h5item]

        if attr.make_dset:
            # hdf5 dataset
            # only the np.ndarray should be non-singular
            if attr.dtype != np.ndarray and h5item.shape != ():
                raise InterfaceError('Found too many elements for attribute ' +
                                     '{}'.format(attr.name))
            if attr.dtype == int or attr.dtype == float or attr.dtype == bool:
                output = np.zeros(())
                h5item.read_direct(output)
                output = attr.dtype(output)
            elif attr.dtype == np.ndarray:
                output = np.zeros(h5item.shape)
                h5item.read_direct(output)
            elif attr.dtype == str:
                # blech
                raise InterfaceError("Don't store strings in datasets!" +
                                     '{}'.format(attr.name))
            else:
                raise InterfaceError('Unknown dtype in dataset for attribute' +
                                     ' {}'.format(attr.name))

        elif attr.is_user_object:
            # user object: recurse
            output = read_object_from_hdf5(
                h5item, h5_to_pydict=h5_to_pydict, verbosity=verbosity)
        elif attr.dtype is np.ndarray:
            output = np.array(h5item)
        else:
            # hdf5 attribute
            output = attr.dtype(h5item)

        return output

    def vprint(stringdata):
        """
        Verbose print
        """
        if verbosity > 0:
            print stringdata

    #
    # ~~~ begin main ~~~
    #
    data_format = check_input(h5group, ext_data_format)
    vprint(' ')
    vprint('Beginning read of {}. h5_to_pydict includes:'.format(str(h5group)))
    for key in h5_to_pydict.keys():
        vprint('    {}'.format(str(key)))

    if h5group in h5_to_pydict:
        # the target of the hard link has already been created
        # this works because (h5groupA == h5groupB) iff they are hardlinks
        #   pointing to the same object in the hdf5 file.
        # (specifically, they are equal but not identical,
        #   i.e. (h5groupA is h5groupB) is false
        #   for hard links pointing to the same object in the hdf5 file)
        vprint('  Found {} in h5_to_pydict! Skipping'.format(str(h5group)))
        output = h5_to_pydict[h5group]
        return output
    else:
        vprint('  {} not in h5_to_pydict. adding and processing...'.format(
               str(h5group)))
        # start ouptput as an empty dictionary
        output = {}
        # add this object to the h5_to_pydict
        h5_to_pydict[h5group] = output

    for attr in data_format:
        vprint('  Attribute {}'.format(attr.name))
        hdf5_type = check_attr(h5group, attr)
        if hdf5_type == 'none':
            output[attr.name] = None
            continue
        elif hdf5_type == 'list':
            i = 0
            output[attr.name] = []
            h5list = h5group[attr.name]
            if attr.make_dset or attr.is_user_object:
                # list elements are stored as hdf5 datasets or hdf5 groups
                while str(i) in h5list:
                    h5item = h5list[str(i)]
                    output[attr.name].append(
                        read_item(attr, h5item, h5_to_pydict=h5_to_pydict))
                    i += 1
            else:
                # list elements are stored as hdf5 attributes
                while str(i) in h5list.attrs:
                    h5item = h5list.attrs[str(i)]
                    output[attr.name].append(
                        read_item(attr, h5item, h5_to_pydict=h5_to_pydict))
                    i += 1

        elif hdf5_type == 'dict':
            output[attr.name] = {}
            # read datasets, groups, and attributes
            for key, h5item in h5group[attr.name].iteritems():
                if key == 'obj_type':
                    continue
                output[attr.name][key] = read_item(
                    attr, h5item, h5_to_pydict=h5_to_pydict)
            for key, h5item in h5group[attr.name].attrs.iteritems():
                if key == 'obj_type':
                    continue
                output[attr.name][key] = read_item(
                    attr, h5item, h5_to_pydict=h5_to_pydict)

        elif hdf5_type == 'single':
            if attr.make_dset or attr.is_user_object:
                h5item = h5group[attr.name]
            else:
                h5item = h5group.attrs[attr.name]
            output[attr.name] = read_item(
                attr, h5item, h5_to_pydict=h5_to_pydict)
        else:
            raise Exception(
                'Unexpected hdf5_type on ' +
                '{}, where did this come from?'.format(attr.name))

    vprint(' ')
    return output


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

    class_name = 'TestIO'

    def __init__(self, data_format, **kwargs):
        """
        Initialize a TestIO object with a user-defined data_format and
        list of keyword arguments.
        """

        self.data_format = data_format
        for key, val in kwargs.iteritems():
            setattr(self, key, val)


def test_IO_singular(filename):
    """
    test all 'singular' data types
    write and read
    """

    # TestIO objects and ClassAttr
    data_format = (ClassAttr('int1', int), ClassAttr('int2', int),
                   ClassAttr('str1', str), ClassAttr('float1', float),
                   ClassAttr('bool1', bool), ClassAttr('array1', np.ndarray))
    array_data = np.array([1.0, 2.1, 3.2])
    t = TestIO(data_format, int1=34, int2=-6, str1='asdf', float1=3.141,
               bool1=False, array1=array_data)
    with h5py.File(filename, 'w') as h5file:
        write_object_to_hdf5(t, h5file, 't')
    with h5py.File(filename, 'r') as h5file:
        t2 = read_object_from_hdf5(h5file['t'], ext_data_format=data_format)
    assert t2['int1'] == 34
    assert t2['int2'] == -6
    assert t2['str1'] == 'asdf'
    assert t2['float1'] == 3.141
    assert t2['bool1'] is False
    assert np.all(t2['array1'] == array_data)
    os.remove(filename)


def test_IO_lists(filename):
    """
    lists and tuples
    is_always_list, is_sometimes_list
    """

    # test list (is_always_list)
    data_format = (ClassAttr('float1', float),
                   ClassAttr('list1', int, is_always_list=True),
                   ClassAttr('str1', str))
    listdata = [1, 3, 5, 7, 9]
    t = TestIO(data_format, float1=-26.3, str1='foo', list1=listdata)
    with h5py.File(filename, 'w') as h5file:
        write_object_to_hdf5(t, h5file, 't')
    with h5py.File(filename, 'r') as h5file:
        t2 = read_object_from_hdf5(h5file['t'], ext_data_format=data_format)
    assert t2['float1'] == -26.3
    assert np.all(t2['list1'] == listdata)
    assert t2['str1'] == 'foo'
    os.remove(filename)

    # test tuple (is_sometimes_list)
    data_format = (ClassAttr('float1', float),
                   ClassAttr('list1', float, is_sometimes_list=True),
                   ClassAttr('str1', str))
    listdata = (1.0, 3.3, 5.1)
    t = TestIO(data_format, float1=-26.3, str1='foo', list1=listdata)
    with h5py.File(filename, 'w') as h5file:
        write_object_to_hdf5(t, h5file, 't')
    with h5py.File(filename, 'r') as h5file:
        t2 = read_object_from_hdf5(h5file['t'], ext_data_format=data_format)
    assert t2['float1'] == -26.3
    assert np.all(t2['list1'] == list(listdata))
    assert t2['str1'] == 'foo'
    os.remove(filename)

    # test is_sometimes_list without a list
    data_format = (ClassAttr('float1', float),
                   ClassAttr('maybelist1', int, is_sometimes_list=True),
                   ClassAttr('str1', str))
    t = TestIO(data_format, float1=-26.3, str1='foo', maybelist1=3)
    with h5py.File(filename) as h5file:
        write_object_to_hdf5(t, h5file, 't')
    with h5py.File(filename, 'r') as h5file:
        t2 = read_object_from_hdf5(h5file['t'], ext_data_format=data_format)
    assert t2['float1'] == -26.3
    assert np.all(t2['maybelist1'] == 3)
    assert t2['str1'] == 'foo'
    os.remove(filename)


def test_IO_dicts(filename):
    """
    dicts
    is_always_dict, is_sometimes_dict
    """

    data_format = (ClassAttr('float1', float),
                   ClassAttr('dict1', str, is_always_dict=True),
                   ClassAttr('str1', str))
    dictdata = {'foo': 'foovalue', 'bar': 'barvalue', 'asdf': 'qwerty'}
    t = TestIO(data_format, float1=-26.3, str1='foo',
               dict1=dictdata)
    with h5py.File(filename, 'w') as h5file:
        write_object_to_hdf5(t, h5file, 't')
    with h5py.File(filename, 'r') as h5file:
        t2 = read_object_from_hdf5(h5file['t'], ext_data_format=data_format)
    assert t2['float1'] == -26.3
    assert t2['str1'] == 'foo'
    assert t2['dict1'] == dictdata
    os.remove(filename)

    # test is_sometimes_dict without a dict
    data_format = (ClassAttr('float1', float),
                   ClassAttr('maybedict1', float, is_sometimes_dict=True),
                   ClassAttr('str1', str))
    t = TestIO(data_format, float1=-26.3, str1='foo', maybedict1=3.5)
    with h5py.File(filename, 'w') as h5file:
        write_object_to_hdf5(t, h5file, 't')
    with h5py.File(filename, 'r') as h5file:
        t2 = read_object_from_hdf5(h5file['t'], ext_data_format=data_format)
    assert t2['float1'] == -26.3
    assert t2['str1'] == 'foo'
    assert t2['maybedict1'] == 3.5
    os.remove(filename)


def test_IO_dsets_none(filename):
    """
    make_dset
    may_be_none
    """

    # test make_dset
    data_format = (ClassAttr('float1', float),
                   ClassAttr('array1', np.ndarray, make_dset=True),
                   ClassAttr('str1', str))
    arraydata = np.array(range(150))
    t = TestIO(data_format, float1=-26.3, str1='foo',
               array1=arraydata)
    with h5py.File(filename, 'w') as h5file:
        write_object_to_hdf5(t, h5file, 't')
    with h5py.File(filename, 'r') as h5file:
        t2 = read_object_from_hdf5(h5file['t'], ext_data_format=data_format)
    assert t2['float1'] == -26.3
    assert t2['str1'] == 'foo'
    assert np.all(t2['array1'] == arraydata)
    os.remove(filename)

    # test may_be_none
    data_format = (ClassAttr('float1', float),
                   ClassAttr('str1', str, may_be_none=True))
    t = TestIO(data_format, float1=-26.3, str1=None)
    with h5py.File(filename, 'w') as h5file:
        write_object_to_hdf5(t, h5file, 't')
    with h5py.File(filename, 'r') as h5file:
        t2 = read_object_from_hdf5(h5file['t'], ext_data_format=data_format)
    assert t2['float1'] == -26.3
    assert t2['str1'] is None
    os.remove(filename)


def test_IO_user_objects(filename):
    """
    single user object
    multi-level user objects
    """

    # don't import at top of file! circular import with evaluation.py
    import evaluation

    # Real Classes:
    # single user-defined object
    alg_results = evaluation.generate_random_alg_results(length=10000)
    with h5py.File(filename, 'w') as h5file:
        write_object_to_hdf5(
            alg_results, h5file, 'alg_results', pyobj_to_h5={})
    with h5py.File(filename, 'r') as h5file:
        ar2 = read_object_from_hdf5(h5file['alg_results'])
    check_alg_results_IO(ar2, alg_results, uncertainty_flag=False)
    os.remove(filename)

    # multi-level object
    alg_results.add_default_uncertainties()
    with h5py.File(filename, 'w') as h5file:
        write_object_to_hdf5(
            alg_results, h5file, 'alg_results', pyobj_to_h5={})
    with h5py.File(filename, 'r') as h5file:
        ar2 = read_object_from_hdf5(h5file['alg_results'])
    check_alg_results_IO(ar2, alg_results, uncertainty_flag=True)
    os.remove(filename)


def check_alg_results_IO(read_dict, orig_obj, uncertainty_flag=False):
    """
    Check that the output from read_object_from_hdf5 (read_dict) has the same
    data as the original object (orig_obj) for an AlgorithmResults object.

    uncertainty_flag: also check alpha_unc and beta_unc
    """

    assert np.all(read_dict['alpha_meas_deg'] == orig_obj.alpha_meas_deg)
    assert np.all(read_dict['alpha_true_deg'] == orig_obj.alpha_true_deg)
    assert np.all(read_dict['beta_meas_deg'] == orig_obj.beta_meas_deg)
    assert np.all(read_dict['beta_true_deg'] == orig_obj.beta_true_deg)
    assert np.all(read_dict['depth_um'] == orig_obj.depth_um)
    assert np.all(read_dict['energy_tot_kev'] == orig_obj.energy_tot_kev)
    assert np.all(read_dict['energy_dep_kev'] == orig_obj.energy_dep_kev)
    assert np.all(read_dict['is_contained'] == orig_obj.is_contained)
    assert read_dict['has_alpha'] == orig_obj.has_alpha
    assert read_dict['has_beta'] == orig_obj.has_beta

    if uncertainty_flag:
        aunc1 = read_dict['alpha_unc']
        aunc2 = orig_obj.alpha_unc
        assert aunc1['angle_type'] == aunc2.angle_type
        assert np.all(aunc1['delta'] == aunc2.delta)
        assert aunc1['n_values'] == aunc2.n_values
        assert aunc1['resolution'] == aunc2.resolution
        assert np.all(aunc1['nhist'] == aunc2.nhist)
        assert np.all(aunc1['xhist'] == aunc2.xhist)

        bunc1 = read_dict['beta_unc']
        bunc2 = orig_obj.beta_unc
        assert bunc1['angle_type'] == bunc2.angle_type
        assert np.all(bunc1['delta'] == bunc2.delta)
        assert bunc1['n_values'] == bunc2.n_values

        for i, unc2 in enumerate(read_dict['uncertainty_list']):
            unc = orig_obj.uncertainty_list[i]
            assert unc2['angle_type'] == unc.angle_type
            assert np.all(unc2['delta'] == unc.delta)
            assert unc2['n_values'] == unc.n_values
            if unc2['angle_type'] == 'alpha':
                assert unc2['resolution'] == unc.resolution
                assert np.all(unc2['nhist'] == unc.nhist)
                assert np.all(unc2['xhist'] == unc.xhist)

            for metricname, metric in unc2['metrics'].iteritems():
                assert metric['axis_max'] == unc.metrics[metricname].axis_max
                assert metric['axis_min'] == unc.metrics[metricname].axis_min
                assert metric['fit_name'] == unc.metrics[metricname].fit_name
                assert metric['name'] == unc.metrics[metricname].name
                assert metric['units'] == unc.metrics[metricname].units
                assert metric['value'] == unc.metrics[metricname].value

        unc = orig_obj.alpha_unc
        iterator = read_dict['alpha_unc']['metrics'].iteritems()
        for metricname, metric in iterator:
            assert metric['axis_max'] == unc.metrics[metricname].axis_max
            assert metric['axis_min'] == unc.metrics[metricname].axis_min
            assert metric['fit_name'] == unc.metrics[metricname].fit_name
            assert metric['name'] == unc.metrics[metricname].name
            assert metric['units'] == unc.metrics[metricname].units
            assert metric['value'] == unc.metrics[metricname].value

        unc = orig_obj.beta_unc
        iterator = read_dict['beta_unc']['metrics'].iteritems()
        for metricname, metric in iterator:
            assert metric['axis_max'] == unc.metrics[metricname].axis_max
            assert metric['axis_min'] == unc.metrics[metricname].axis_min
            assert metric['fit_name'] == unc.metrics[metricname].fit_name
            assert metric['name'] == unc.metrics[metricname].name
            assert metric['units'] == unc.metrics[metricname].units
            assert metric['value'] == unc.metrics[metricname].value


def test_IO_obj_dict(filename):
    """
    test obj_dict hardlink capability
    """

    import evaluation

    # first, check a single object which is listed in obj_dict
    alg_results = evaluation.generate_random_alg_results(length=10000)
    alg_results.parent = [alg_results]
    with h5py.File(filename, 'w') as h5file:
        write_object_to_hdf5(
            alg_results, h5file, 'alg_results', pyobj_to_h5={})
    with h5py.File(filename, 'r') as h5file:
        ar2 = read_object_from_hdf5(
            h5file['alg_results'], h5_to_pydict={})
    assert ar2['parent'][0] is ar2
    os.remove(filename)

    # check alpha_unc and beta_unc
    alg_results.add_default_uncertainties()
    alg_results.parent = [alg_results]
    with h5py.File(filename, 'w') as h5file:
        write_object_to_hdf5(
            alg_results, h5file, 'alg_results', pyobj_to_h5={})
    with h5py.File(filename, 'r') as h5file:
        ar2 = read_object_from_hdf5(
            h5file['alg_results'], h5_to_pydict={})
    assert ar2['parent'][0] is ar2
    assert ar2['alpha_unc'] is ar2['uncertainty_list'][0]
    assert ar2['beta_unc'] is ar2['uncertainty_list'][1]
    os.remove(filename)

    # check obj_dict with multiple objects written to the same file
    # (has to be written in the same file session)
    ar1 = evaluation.generate_random_alg_results(length=10000)
    ar1.add_default_uncertainties()
    ar2 = evaluation.generate_random_alg_results(length=1000)
    ar2.add_default_uncertainties()
    obj_dict = {}
    with h5py.File(filename, 'a') as h5file:
        write_object_to_hdf5(ar1, h5file, 'ar1', pyobj_to_h5=obj_dict)
        write_object_to_hdf5(ar2, h5file, 'ar2', pyobj_to_h5=obj_dict)
        # should be hardlinked
        write_object_to_hdf5(ar1, h5file, 'ar3', pyobj_to_h5=obj_dict)
    with h5py.File(filename, 'r') as h5file:
        ar1r = read_object_from_hdf5(h5file['ar1'])
        ar2r = read_object_from_hdf5(h5file['ar2'])
        ar3r = read_object_from_hdf5(h5file['ar3'])
    check_alg_results_IO(ar1r, ar1, uncertainty_flag=True)
    check_alg_results_IO(ar2r, ar2, uncertainty_flag=True)
    check_alg_results_IO(ar3r, ar1, uncertainty_flag=True)
    # this is the hardlink test:
    assert ar1r is ar3r


def test_IO_overwrite(filename):
    """
    test the ability to overwrite objects in an existing HDF5 file
    """

    import evaluation

    # simple overwrite
    alg_results = evaluation.generate_random_alg_results(length=10000)
    alg_results.add_default_uncertainties()
    with h5py.File(filename, 'a') as h5file:
        write_object_to_hdf5(
            alg_results, h5file, 'alg_results', pyobj_to_h5={})
    with h5py.File(filename, 'a') as h5file:
        write_object_to_hdf5(
            alg_results, h5file, 'alg_results', pyobj_to_h5={})
    with h5py.File(filename, 'r') as h5file:
        ar2 = read_object_from_hdf5(h5file['alg_results'])
    check_alg_results_IO(ar2, alg_results, uncertainty_flag=True)
    os.remove(filename)

    # writing two objects to the same file
    ar1 = evaluation.generate_random_alg_results(length=10000)
    ar1.add_default_uncertainties()
    ar2 = evaluation.generate_random_alg_results(length=1000)
    ar2.add_default_uncertainties()
    with h5py.File(filename, 'a') as h5file:
        write_object_to_hdf5(ar1, h5file, 'ar1', pyobj_to_h5={})
        write_object_to_hdf5(ar2, h5file, 'ar2', pyobj_to_h5={})
    with h5py.File(filename, 'r') as h5file:
        ar1r = read_object_from_hdf5(h5file['ar1'])
        ar2r = read_object_from_hdf5(h5file['ar2'])
    check_alg_results_IO(ar1r, ar1, uncertainty_flag=True)
    check_alg_results_IO(ar2r, ar2, uncertainty_flag=True)

    # overwriting just one of the two objects
    ar3 = evaluation.generate_random_alg_results(length=1000)
    ar3.add_default_uncertainties()
    with h5py.File(filename, 'a') as h5file:
        write_object_to_hdf5(ar3, h5file, 'ar1', pyobj_to_h5={})
    with h5py.File(filename, 'r') as h5file:
        ar1r = read_object_from_hdf5(h5file['ar1'])
        ar2r = read_object_from_hdf5(h5file['ar2'])
    check_alg_results_IO(ar1r, ar3, uncertainty_flag=True)
    check_alg_results_IO(ar2r, ar2, uncertainty_flag=True)


def test_write_objects_to_hdf5():
    """
    test the multiple-object form of writing
    """

    import evaluation

    filebase = ''.join(chr(i) for i in np.random.randint(97, 122, size=(8,)))
    filename = '.'.join([filebase, 'h5'])

    # h5file provided
    # single object
    ar = evaluation.generate_random_alg_results(length=1000)
    ar.add_default_uncertainties()
    with h5py.File(filename, 'a') as h5file:
        write_objects_to_hdf5(h5file, ar=ar)
    with h5py.File(filename, 'r') as h5file:
        ar_read = read_object_from_hdf5(h5file['ar'])
    check_alg_results_IO(ar_read, ar, uncertainty_flag=True)
    os.remove(filename)

    # h5file provided
    # multiple objects
    ar1 = evaluation.generate_random_alg_results(length=1000)
    ar1.add_default_uncertainties()
    ar2 = evaluation.generate_random_alg_results(length=2000)
    ar2.add_default_uncertainties()
    ar3 = evaluation.generate_random_alg_results(length=3000)
    ar3.add_default_uncertainties()
    with h5py.File(filename, 'a') as h5file:
        filename_written = write_objects_to_hdf5(
            h5file,
            ar1=ar1, ar2=ar2, ar3=ar3, aunc=ar1.alpha_unc)
    assert filename_written == filename
    with h5py.File(filename, 'r') as h5file:
        ar1_read = read_object_from_hdf5(h5file['ar1'])
        ar2_read = read_object_from_hdf5(h5file['ar2'])
        ar3_read = read_object_from_hdf5(h5file['ar3'])
        aunc = read_object_from_hdf5(h5file['aunc'])
    check_alg_results_IO(ar1_read, ar1, uncertainty_flag=True)
    check_alg_results_IO(ar2_read, ar2, uncertainty_flag=True)
    check_alg_results_IO(ar3_read, ar3, uncertainty_flag=True)
    # check hard link across multiple write calls (within a file session)
    assert aunc is ar1_read['alpha_unc']

    # filename provided, including extension (single object)
    filename_written = write_objects_to_hdf5(filename, ar=ar)
    assert filename_written == filename
    with h5py.File(filename, 'r') as h5file:
        ar_read = read_object_from_hdf5(h5file['ar'])
    check_alg_results_IO(ar_read, ar, uncertainty_flag=True)
    os.remove(filename)

    # filename provided as *.hdf5
    filename_hdf5 = '.'.join([filebase, 'hdf5'])
    filename_written = write_objects_to_hdf5(filename_hdf5, ar=ar)
    assert filename_written == filename_hdf5
    with h5py.File(filename_hdf5, 'r') as h5file:
        ar_read = read_object_from_hdf5(h5file['ar'])
    check_alg_results_IO(ar_read, ar, uncertainty_flag=True)
    os.remove(filename_hdf5)

    # filename provided without extension. check that extension is added
    filename_written = write_objects_to_hdf5(filebase, ar=ar)
    assert filename_written == filename
    with h5py.File(filename, 'r') as h5file:
        ar_read = read_object_from_hdf5(h5file['ar'])
    check_alg_results_IO(ar_read, ar, uncertainty_flag=True)
    os.remove(filename)


if __name__ == '__main__':
    """
    Run tests.
    """

    # save time for routine testing
    run_file_tests = False

    test_G4Track()
    test_Track()
    test_TrackExceptions()
    test_AlgorithmOutput()

    filebase = ''.join(chr(i) for i in np.random.randint(97, 122, size=(8,)))
    filename = '.'.join([filebase, 'h5'])

    try:
        test_IO_singular(filename)
        test_IO_lists(filename)
        test_IO_dicts(filename)
        test_IO_dsets_none(filename)
        test_IO_user_objects(filename)
        test_IO_obj_dict(filename)
        test_IO_overwrite(filename)
        test_write_objects_to_hdf5()
    finally:
        # if any exceptions are raised in the test, the file will not have
        #   been deleted by os.remove(). So try it here.
        #   (too lazy to actually check whether it's still there or not,
        #    so try/except))
        try:
            os.remove(filename)
        except OSError:
            pass

    import sys

    try:
        h5initial = h5py.File('MultiAngle_HT_11_12.h5', 'r')
    except IOError:
        print('Skipping file tests')
        sys.exit()

    if not run_file_tests:
        print('Skipping file tests')
        sys.exit()

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
