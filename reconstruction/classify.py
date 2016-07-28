# classify a g4track

import numpy as np
import matplotlib.pyplot as plt

from etrack.reconstruction.trackdata import Track, G4Track
import etrack.reconstruction.evaluation as ev

BIG_STEP_UM = 1.0   # micron


class Classifier(object):
    """
    Object to handle auto classifying (ground truth) a Monte Carlo track
    """

    def __init__(self, g4track):
        assert isinstance(g4track, G4Track), "Not a G4Track object"
        self.g4track = g4track

        assert g4track.x.shape[0] == 3, "x has funny shape"
        self.x = g4track.x

        self.E = g4track.dE

    def classify(self, scatterlen=50, overlapdist=50):
        """
        Classify the Monte Carlo track as either:
          'good',
          early scatter 'scatter',
          overlapping the initial end 'overlap',
          or no result, None.

        Optional input args:
          scatterlen: length from initial end, in um, to look for
            high-angle scatter. (default 50 um)
          overlapdist: if points are
        """

        self.flag_backsteps()
        self.flag_newparticle()
        self.reorder_backsteps()

    def flag_backsteps(self):
        """
        Look for backwards steps (from misordering in DiffuseTrack).
        Make an attribute that flags them.

        Backwards steps are marked by <0.5um step and >90 degree from previous
        step direction.
        """

        self.dx = self.x[:, 1:] - self.x[:, :-1]
        # step lengths
        self.d = np.linalg.norm(self.dx, axis=0)
        # step directions
        self.stepdirs = self.normalize_steps(self.dx, d=self.d)
        # cos(difference in direction): dot product.
        # (1 = same direction, 0 = 90 degree turn, -1 = opposite direction)
        ddir = np.sum(self.stepdirs[:, 1:] * self.stepdirs[:, :-1], axis=0)
        self.ddir = np.concatenate(([0], ddir))
        # make it the same size as dx.
        # ddir[i] is the difference between stepdirs[i-1] and stepdirs[i]

        self.dx_backward_flag = (self.d < BIG_STEP_UM / 2) & (self.ddir < 0)

    def flag_newparticle(self):
        """
        Look for particle transitions - jumping to a new electron ID in Geant4.

        Marked by >1.5um step.
        """

        if not hasattr(self, 'd'):
            self.dx = self.x[:, 1:] - self.x[:, :-1]
            self.d = np.linalg.norm(self.dx, axis=0)

        self.dx_newparticle_flag = (self.d > BIG_STEP_UM * 1.5)

    def reorder_backsteps(self):
        """
        Reorder positions and energies according to dx_backward_flag.
        The before and after positions of the backstep need to be switched.
        """

        # only operate on the first particle
        new_particle_ind = np.nonzero(self.dx_newparticle_flag)[0][0]
        assert not self.dx_backward_flag[0], 'First step is backward??'
        # First, check that no backsteps are consecutive
        consecutive = (self.dx_backward_flag[:new_particle_ind - 1] &
                       self.dx_backward_flag[1:new_particle_ind])
        assert not np.any(consecutive), 'Found consecutive backsteps'
        consecutive_forward = (
            np.logical_not(self.dx_backward_flag[:new_particle_ind - 1]) &
            np.logical_not(self.dx_backward_flag[1:new_particle_ind]))
        print("{} consecutive forward steps".format(
            np.sum(consecutive_forward)))

        for ind in np.nonzero(self.dx_backward_flag)[0]:
            if ind >= new_particle_ind:
                break
            # swap before and after. dx is indexed as the delta.
            (self.x[:, ind], self.x[:, ind + 1]) = (
                self.x[:, ind + 1], self.x[:, ind])
            (self.E[ind], self.E[ind + 1]) = (self.E[ind + 1], self.E[ind])

        self.flag_backsteps()
        self.flag_newparticle()

    def normalize_steps(self, dx, d=None):
        """
        normalize steps into unit vectors
        dx.shape should be (3, n)
        """

        assert dx.shape[0] == 3, "dx has funny shape in normalize_steps"
        if d is None:
            d = np.linalg.norm(dx, axis=0)

        norm_dx = dx / d
        return norm_dx

    def round_steplength(self, d):
        """
        round to nearest half-big-step (nearest 0.5 um)
        """
        return np.round(d / (BIG_STEP_UM / 2)) * (BIG_STEP_UM / 2)
