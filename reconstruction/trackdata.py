#!/usr/bin/python

from __future__ import print_function

import numpy as np
import h5py
import datetime
import ipdb as pdb

import hybridtrack
import dataformats
import trackio


##############################################################################
#                                  G4Track                                   #
##############################################################################


class G4Track(object):
    """
    Electron track from Geant4.
    """

    __version__ = '0.2'
    class_name = 'G4Track'
    data_format = dataformats.get_format(class_name)

    # a lot more attributes could be added here...
    attr_list = (
        'x',
        'dE',
        'x0',
        'alpha_deg',
        'beta_deg',
        'first_step_vector',
        'energy_tot_kev',
        'energy_dep_kev',
        'energy_esc_kev',
        'energy_xray_kev',
        'energy_brems_kev',
        'depth_um',
        'is_contained',
    )

    def __init__(self, matrix=None, **kwargs):
        """
        Construct G4Track object.

        If matrix is supplied and other quantities are not, then the other
        quantities will be calculated using the matrix (not implemented yet).

          matrix
          (see attr_list class variable)
        """

        self.matrix = matrix

        for attr in self.attr_list:
            if attr in kwargs:
                setattr(self, attr, kwargs[attr])
            else:
                setattr(self, attr, None)

        if matrix is not None and (
                'x' not in kwargs or
                'dE' not in kwargs or
                'energy_tot_kev' not in kwargs or
                'energy_dep_kev' not in kwargs or
                'energy_esc_kev' not in kwargs or
                'depth_um' not in kwargs or
                'is_contained' not in kwargs):
            pass
            # self.measure_quantities()

    @classmethod
    def from_h5matlab(cls, evt):
        """
        Construct a G4Track instance from an event in an HDF5 file.

        The format of the HDF5 file is 'matlab', a.k.a. the more complete mess
        that Brian made in December 2015.
        """

        if evt.attrs['multiplicity'] > 1:
            # at this time, I am not handling multiple-scattered photons
            return None

        data = evt['trackM']
        matrix = np.zeros(data.shape)
        data.read_direct(matrix)
        matrix = np.array(matrix)

        cheat = evt['cheat']['0']

        # h5 attributes
        kwargs = {
            'energy_tot_kev': cheat.attrs['Etot'],
            'energy_dep_kev': cheat.attrs['Edep'],
            'energy_esc_kev': evt.attrs['Eesc'],
            'energy_xray_kev': cheat.attrs['Exray'],
            'energy_brems_kev': cheat.attrs['Ebrems'],
            'x0': cheat.attrs['x0'],
            'first_step_vector': cheat.attrs['firstStepVector'],
            'alpha_deg': cheat.attrs['alpha'],
            'beta_deg': cheat.attrs['beta']
        }

        # h5 datasets
        data = cheat['x']
        x = np.zeros(data.shape)
        data.read_direct(x)
        kwargs['x'] = x

        data = cheat['dE']
        dE = np.zeros(data.shape)
        data.read_direct(dE)
        kwargs['dE'] = dE

        g4track = G4Track(matrix=matrix, **kwargs)
        return g4track

    @classmethod
    def from_pydict(cls, read_dict, pydict_to_pyobj=None):
        """
        Initialize a G4Track object from the dictionary returned by
        trackio.read_object_from_hdf5().
        """

        if pydict_to_pyobj is None:
            pydict_to_pyobj = {}

        if id(read_dict) in pydict_to_pyobj:
            return pydict_to_pyobj[id(read_dict)]

        other_attrs = ('matrix',)
        all_attrs = other_attrs + cls.attr_list

        kwargs = {}
        for attr in all_attrs:
            kwargs[attr] = read_dict.get(attr)
            # read_dict.get() defaults to None, although this actually
            #   shouldn't be needed since read_object_from_hdf5 adds Nones
        constructed_object = cls(**kwargs)

        # add entry to pydict_to_pyobj
        pydict_to_pyobj[id(read_dict)] = constructed_object

        return constructed_object

    @classmethod
    def from_hdf5(cls, h5group, h5_to_pydict=None, pydict_to_pyobj=None):
        """
        Initialize a G4Track instance from an HDF5 group.
        """

        if h5_to_pydict is None:
            h5_to_pydict = {}
        if pydict_to_pyobj is None:
            pydict_to_pyobj = {}

        read_dict = trackio.read_object_from_hdf5(
            h5group, h5_to_pydict=h5_to_pydict)

        constructed_object = cls.from_pydict(
            read_dict, pydict_to_pyobj=pydict_to_pyobj)

        return constructed_object

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

    attr_list = (
        'is_modeled',
        'is_measured',
        'pixel_size_um',
        'noise_ev',
        'g4track',
        'energy_kev',
        'x_offset_pix',
        'y_offset_pix',
        'timestamp',
        'shutter_ind',
        'label',
    )

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

        if (timestamp is not None and timestamp != 'None' and
                not isinstance(timestamp, datetime.datetime)):
            raise InputError('timestamp should be a datetime object')
        self.timestamp = str(timestamp)

        if label is not None:
            label = str(label)
        self.label = label

    @classmethod
    def from_h5matlab(cls, pixnoise, g4track=None):
        """
        Construct a Track object from one pixelsize/noise of an event in an
        HDF5 file.

        HDF5 file is the December 2015 format from MATLAB.
        """

        errorcode = pixnoise.attrs['errorcode']
        if errorcode != 0:
            # for now, only accept tracks without errors
            return errorcode
        # algorithm_error = (errorcode == 4 or errorcode == 5)
        # good_algorithm = (errorcode == 0)
        # has_algorithm = good_algorithm or algorithm_error
        # has_ridge = good_algorithm
        # has_measurement = good_algorithm
        # has_multiple_tracks = (errorcode == 2 or errorcode == 6)
        #
        # do_pixnoise = true
        # check_pixsize = good_algorithm or has_multiple_tracks
        # check_noise = has_multiple_tracks
        # do_img = has_algorithm
        #
        # do_EtotTind = has_algorithm
        # do_nends = good_algorithm or errorcode == 4
        #
        # do_T = has_multiple_tracks
        # do_edgesegments = good_algorithm
        # do_ridge = has_ridge
        # do_measurement = has_measurement

        # track info (not algorithm info)
        data = pixnoise['img']
        img = np.zeros(data.shape)
        data.read_direct(img)

        kwargs = {}
        kwargs['is_modeled'] = True
        kwargs['g4track'] = g4track
        kwargs['pixel_size_um'] = pixnoise.attrs['pixel_size_um']
        kwargs['noise_ev'] = pixnoise.attrs['noise_ev']
        kwargs['energy_kev'] = pixnoise.attrs['Etot']

        track = Track(img, **kwargs)

        # algorithm info
        alpha = pixnoise.attrs['alpha']
        beta = pixnoise.attrs['beta']
        info = MatlabAlgorithmInfo.from_h5pixnoise(pixnoise)

        track.add_algorithm('matlab HT v1.5',
                            alpha_deg=alpha, beta_deg=beta,
                            info=info)
        return track

    @classmethod
    def from_pydict(cls, read_dict, pydict_to_pyobj=None):
        """
        Initialize a Track object from the dictionary returned by
        trackio.read_object_from_hdf5().
        """

        if pydict_to_pyobj is None:
            pydict_to_pyobj = {}

        if id(read_dict) in pydict_to_pyobj:
            return pydict_to_pyobj[id(read_dict)]

        other_attrs = ('image', 'algorithms')
        all_attrs = other_attrs + cls.attr_list

        kwargs = {}
        # keep algorithms in a separate dict because they do not go in __init__
        algorithms = {}
        for attr in all_attrs:
            if attr == 'g4track' and read_dict.get(attr) is not None:
                kwargs[attr] = G4Track.from_pydict(
                    read_dict[attr], pydict_to_pyobj=pydict_to_pyobj)
                # if g4track *is* None, then it gets assigned in "else" below
            elif attr == 'algorithms':
                if read_dict.get(attr) is not None:
                    for key, val in read_dict[attr].iteritems():
                        algorithms[key] = AlgorithmOutput.from_pydict(
                            read_dict[attr][key],
                            pydict_to_pyobj=pydict_to_pyobj)
                # else, algorithms is still {} as it should be
            else:
                kwargs[attr] = read_dict.get(attr)
            # read_dict.get() defaults to None, although this actually
            #   shouldn't be needed since read_object_from_hdf5 adds Nones

        image = kwargs.pop('image')
        constructed_object = cls(image, **kwargs)
        for key, algoutput in algorithms.iteritems():
            alpha = algoutput.alpha_deg
            beta = algoutput.beta_deg
            info = algoutput.info
            constructed_object.add_algorithm(key, alpha, beta, info)

        # add entry to pydict_to_pyobj
        pydict_to_pyobj[id(read_dict)] = constructed_object

        return constructed_object

    @classmethod
    def from_hdf5(cls, h5group, h5_to_pydict=None, pydict_to_pyobj=None):
        """
        Initialize a Track instance from an HDF5 group.
        """

        if h5_to_pydict is None:
            h5_to_pydict = {}
        if pydict_to_pyobj is None:
            pydict_to_pyobj = {}

        read_dict = trackio.read_object_from_hdf5(
            h5group, h5_to_pydict=h5_to_pydict)

        constructed_object = cls.from_pydict(
            read_dict, pydict_to_pyobj=pydict_to_pyobj)

        return constructed_object

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


class MatlabAlgorithmInfo(object):
    """
    An empty container to store attributes loaded from the Matlab algorithm.
    """

    __version__ = '0.1'
    class_name = 'MatlabAlgorithmInfo'
    data_format = dataformats.get_format(class_name)

    attr_list = (
        'Tind',
        'lt',
        'n_ends',
        'Eend',
        'alpha',
        'beta',
        'dalpha',
        'dbeta',
        'edgesegments_energies_kev',
        'edgesegments_coordinates_pix',
        'edgesegments_chosen_index',
        'edgesegments_start_coordinates_pix',
        'edgesegments_start_direction_indices',
        'edgesegments_low_threshold_used',
        'dedx_ref',
        'dedx_meas',
        'measurement_start_ind',
        'measurement_end_ind',
    )
    data_list = (
        'thinned_img',
        'x',
        'y',
        'w',
        'a0',
        'dE',
    )

    def __init__(self, **kwargs):
        """
        Shouldn't need to call this -- use from_h5pixnoise or from_pydict
        """

        for attr in kwargs:
            setattr(self, attr, kwargs[attr])

    @classmethod
    def from_h5pixnoise(cls, pixnoise):
        """
        Initialize MatlabAlgorithmInfo with all the attributes from a
        successful HybridTrack algorithm.

        pixnoise is the h5 group.

        This goes with the _h5matlab format.
        """

        kwargs = {}
        for attr in cls.attr_list:
            if attr in pixnoise.attrs:
                kwargs[attr] = pixnoise.attrs[attr]
            else:
                raise InputError(
                    'Missing h5 attribute: {} in {}'.format(
                        attr, str(pixnoise)))
        for attr in cls.data_list:
            if attr in pixnoise:
                kwargs[attr] = pixnoise[attr]
            else:
                raise InputError(
                    'Missing h5 dataset: {} in {}'.format(
                        attr, str(pixnoise)))
        return cls(**kwargs)

    @classmethod
    def from_pydict(cls, read_dict, pydict_to_pyobj=None):
        """
        Initialize a MatlabAlgorithmInfo object from the dictionary returned by
        trackio.read_object_from_hdf5().
        """

        if pydict_to_pyobj is None:
            pydict_to_pyobj = {}

        if id(read_dict) in pydict_to_pyobj:
            return pydict_to_pyobj[id(read_dict)]

        all_attrs = cls.attr_list

        kwargs = {}
        for attr in all_attrs:
            kwargs[attr] = read_dict.get(attr)
            # read_dict.get() defaults to None, although this actually
            #   shouldn't be needed since read_object_from_hdf5 adds Nones
        constructed_object = cls(**kwargs)

        # add entry to pydict_to_pyobj
        pydict_to_pyobj[id(read_dict)] = constructed_object

        return constructed_object

    @classmethod
    def from_hdf5(cls, h5group,
                  h5_to_pydict=None, pydict_to_pyobj=None):
        """
        Initialize a MatlabAlgorithmInfo object from an HDF5 group.
        """

        if h5_to_pydict is None:
            h5_to_pydict = {}
        if pydict_to_pyobj is None:
            pydict_to_pyobj = {}

        read_dict = trackio.read_object_from_hdf5(
            h5group, h5_to_pydict=h5_to_pydict)

        constructed_object = cls.from_pydict(
            read_dict, pydict_to_pyobj=pydict_to_pyobj)

        return constructed_object


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

    __version__ = '0.2'
    class_name = 'AlgorithmOutput'

    def __init__(self, alg_name, alpha_deg, beta_deg, info=None):
        self.alpha_deg = alpha_deg
        self.beta_deg = beta_deg
        self.alg_name = alg_name
        self.info = info

        # customize data_format based on which algorithm type
        if self.alg_name.startswith('matlab'):
            self.data_format = dataformats.get_format('AlgorithmOutputMatlab')
        else:
            self.data_format = dataformats.get_format(self.class_name)

    @classmethod
    def from_pydict(cls, read_dict, pydict_to_pyobj=None):
        """
        Initialize an AlgorithmOutput object from the dictionary returned by
        trackio.read_object_from_hdf5().
        """

        if pydict_to_pyobj is None:
            pydict_to_pyobj = {}

        if id(read_dict) in pydict_to_pyobj:
            return pydict_to_pyobj[id(read_dict)]

        # handle "info" later
        all_attrs = ('alg_name', 'alpha_deg', 'beta_deg')

        kwargs = {}
        for attr in all_attrs:
            kwargs[attr] = read_dict[attr]

        if kwargs['alg_name'].startswith('matlab') and 'info' in read_dict:
            kwargs['info'] = MatlabAlgorithmInfo.from_pydict(
                read_dict['info'], pydict_to_pyobj=pydict_to_pyobj)

        constructed_object = cls(**kwargs)

        # add entry to pydict_to_pyobj
        pydict_to_pyobj[id(read_dict)] = constructed_object

        return constructed_object


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


##############################################################################
#                                  Testing                                   #
##############################################################################


def test_G4Track():
    """
    """

    def test_G4Track_IO():
        """
        """

        # TODO
        print('test_G4Track_IO not implemented yet!')
        pass

        test_G4Track_from_pydict()
        test_G4Track_from_hdf5()

    def test_G4Track_from_pydict():
        pass

    def test_G4Track_from_hdf5():
        pass

    # test_G4Track() main

    G4Track(matrix=None, alpha_deg=132.5, beta_deg=-43.5, energy_tot_kev=201.2)

    # G4Track(matrix=test_matrix())

    test_G4Track_IO()


def test_Track():
    """
    """

    def test_Track_read(track):
        # test Track data format
        import trackio
        import os

        filebase = ''.join(
            chr(i) for i in np.random.randint(97, 122, size=(8,)))
        filename = '.'.join([filebase, 'h5'])
        with h5py.File(filename, 'a') as h5file:
            trackio.write_object_to_hdf5(
                track, h5file, 'track')
        with h5py.File(filename, 'r') as h5file:
            track2 = trackio.read_object_from_hdf5(
                h5file['track'])

        assert track2['is_modeled'] == track.is_modeled
        assert track2['pixel_size_um'] == track.pixel_size_um
        assert track2['noise_ev'] == track.noise_ev
        assert track2['label'] == track.label
        assert track2['energy_kev'] == track.energy_kev
        assert np.all(track2['image'] == track.image)

        assert track2['algorithms']['python HT v1.5']['alpha_deg'] == 120.5
        assert track2['algorithms']['python HT v1.5']['beta_deg'] == 43.5

        track3 = test_Track_from_pydict(track, track2)

        test_Track_from_hdf5()

        os.remove(filename)

    def test_Track_from_pydict(track, track2):
        """
        track is the original generated Track object
        track2 is the pydict from file

        Returns the track object constructed from pydict.
        """
        track3 = Track.from_pydict(track2)
        assert track3.is_modeled == track.is_modeled
        assert track3.pixel_size_um == track.pixel_size_um
        assert track3.noise_ev == track.noise_ev
        assert track3.label == track.label
        assert track3.energy_kev == track.energy_kev
        assert np.all(track3.image == track.image)

        assert track3.algorithms['python HT v1.5'].alpha_deg == 120.5
        assert track3.algorithms['python HT v1.5'].beta_deg == 43.5

        return track3

    def test_Track_from_hdf5():
        """
        as previously, but test the Track.from_hdf5() constructor.
        """

        # TODO
        print('test_Track_from_hdf5 not implemented yet!')
        pass

    # test_Track() main

    image = hybridtrack.test_input()

    track = Track(image, is_modeled=True, pixel_size_um=10.5, noise_ev=0.0,
                  label='MultiAngle', energy_kev=np.sum(image))
    options, info = hybridtrack.reconstruct(image)

    track.add_algorithm('python HT v1.5', 120.5, 43.5, info=info)

    test_Track_read(track)


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


def test_AlgorithmOutput():
    """
    """

    AlgorithmOutput('matlab HT v1.5', 120.5, 43.5)


def test_h5matlab(h5file):
    """
    includes G4Track and Track.

    tested with this file:

    loadpath = ('/home/plimley/Documents/MATLAB/data/Electron Track/' +
                'algorithms/results/2013sep binned')
    loadname = 'MultiAngle_HT_20_2.h5'
    """

    evt = h5file['00000']
    g4track = G4Track.from_h5matlab(evt)
    pixnoise = {}
    for key in evt:
        if key.startswith('pix'):
            pixnoise[key] = Track.from_h5matlab(
                evt[key], g4track=g4track)
            if isinstance(pixnoise[key], Track):
                assert pixnoise[key].is_modeled
            else:
                # it is just a numeric errorcode,
                #   because from_h5matlab doesn't handle errorcodes
                pass

    # TODO: assertions!


def main():
    """
    Run tests.
    """

    import os

    # save time for routine testing
    # run_file_tests = True

    test_G4Track()
    test_Track()
    test_TrackExceptions()
    test_AlgorithmOutput()

    try:
        loadpath = ('/home/plimley/Documents/MATLAB/data/Electron Track/' +
                    'algorithms/results/2013sep binned')
        loadname = 'MultiAngle_HT_20_2.h5'
        filename = os.path.join(loadpath, loadname)

        with h5py.File(filename, 'r') as h5file:
            print('Running h5matlab file test')
            test_h5matlab(h5file)

        # # debug version: no auto close file
        # h5file = h5py.File(filename, 'r')
        # print('Running h5matlab file test')
        # test_h5matlab(h5file)
        # h5file.close()
    except IOError:
        print('Skipping h5matlab file test')


if __name__ == '__main__':
    main()
