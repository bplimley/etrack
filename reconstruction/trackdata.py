#!/usr/bin/python

import numpy as np
# import h5py
import datetime

import hybridtrack


class G4Track(object):
    """
    Electron track from Geant4.
    """

    def __init__(self, matrix=None, alpha_deg=None, beta_deg=None,
                 energy_tot_kev=None, energy_dep_kev=None, energy_esc_kev=None,
                 x=None, dE=None):
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
        """

        self.matrix = matrix

        if matrix is not None and (
                x is None or dE is None or
                energy_tot_kev is None or energy_dep_kev is None or
                energy_esc_kev is None or x is None or dE is None):
            self.measure_quantities()
        else:
            self.alpha_deg = alpha_deg
            self.beta_deg = beta_deg
            self.energy_tot_kev = energy_tot_kev
            self.energy_dep_kev = energy_dep_kev
            self.energy_esc_kev = energy_esc_kev
            self.x = x
            self.dE = dE

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
            matrix=None, alpha_deg=alpha, beta_deg=beta,
            energy_tot_kev=energy_tot, energy_dep_kev=energy_dep,
            energy_esc_kev=None, x=None, dE=None)

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
        """

        # TODO
        raise NotImplementedError("haven't written this yet")

        if self.matrix is None:
            raise RuntimeError('measure_quantities needs a geant4 matrix')

    # TODO:
    # add interface for referencing Track objects associated with this G4Track
    #
    # dictionary-style or ...?
    #
    # see http://www.diveintopython3.net/special-method-names.html


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
            raise RuntimeError('Please specify modeled or measured!')
        elif ((is_modeled is True and is_measured is True) or
              (is_modeled is True and is_experimental is True)):
            raise RuntimeError('Track cannot be both modeled and measured!')
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
                raise RuntimeError('timestamp should be a datetime object')
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
            if fieldname.starts_with('pix'):
                tracks[fieldname] = cls.from_h5initial_one(evt, fieldname,
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
                      is_modeled=True, image=image, pixel_size_um=pix,
                      noise_ev=noise, g4track=g4track,
                      label='MultiAngle h5 initial')
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
            raise RuntimeError(alg_name + " already in algorithms")
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

    test_AlgorithmOutput()
    test_Track()
    test_TrackExceptions()
    test_G4Track()
