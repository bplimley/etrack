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
        self.x = np.copy(g4track.x)

        self.E = np.copy(g4track.dE.flatten())

    def classify(self, scatterlen_um=25, overlapdist_um=40, verbose=False):
        """
        Classify the Monte Carlo track as either:
          'good',
          early scatter 'scatter',
          overlapping the initial end 'overlap',
          or no result, None.

        Optional input args:
          scatterlen_um: length from initial end, in um, to look for
            high-angle scatter. (default 50 um)
          overlapdist_um: if points are
        """

        self.scatterlen_um = scatterlen_um
        self.overlapdist_um = overlapdist_um

        self.flag_newparticle()

        self.check_early_scatter(v=verbose)
        self.check_overlap()

    def flag_newparticle(self):
        """
        Look for particle transitions - jumping to a new electron ID in Geant4.

        Marked by >1.5um step.
        """

        if not hasattr(self, 'd'):
            self.dx = self.x[:, 1:] - self.x[:, :-1]
            self.d = np.linalg.norm(self.dx, axis=0)

        self.dx_newparticle_flag = (self.d > BIG_STEP_UM * 1.5)

    def check_early_scatter(self, v=False):
        """
        look for a >90 degree direction change within the first scatterlen_um
        of track.

        track should already be fixed using reorder_backsteps().
        """

        angle_threshold_deg = 90
        angle_threshold_rad = angle_threshold_deg / 180 * np.pi
        angle_threshold_cos = np.cos(angle_threshold_rad)

        self.scatterlen_steps = np.round(self.scatterlen_um / BIG_STEP_UM * 2)
        self.x2 = self.x[:,:self.scatterlen_steps:2]
        self.dx2 = self.x2[:, 1:] - self.x2[:, :-1]
        self.dx2norm = self.normalize_steps(self.dx2)
        self.ddir2 = np.sum(self.dx2norm[:, 1:] * self.dx2norm[:, :-1], axis=0)

        self.early_scatter = np.any(self.ddir2 < angle_threshold_cos)

        if self.early_scatter and v:
            print('Early scatter!')

    def check_overlap(self):
        """
        look for a section of track overlapping with the initial scatterlen_um
        of track.
        """

        self.overlap = False    # until shown otherwise

        # see which points are within overlapdist_um of these points
        # distance matrix: distance of all points from each of the first 50

        # arrays of dimensions (self.x.shape[1], self.numsteps)
        all_x, init_x = np.meshgrid(self.x[0, :self.numsteps], self.x[0, :])
        all_y, init_y = np.meshgrid(self.x[1, :self.numsteps], self.x[1, :])

        dist_matrix = (all_x - init_x)**2 + (all_y - init_y)**2
        dist_vector = np.min(dist_matrix, axis=1)

        # don't bother with sqrt
        dist_threshold = self.overlapdist_um**2
        too_close = (dist_vector < dist_threshold) + 0  # as an int

        # the initial segment, and some points after it, are obviously
        # going to be too close.

        # first try: if at least overlapdist of consecutive points are
        # too_close, then classify the track as overlapping
        #   i.e. at least overlapdist consecutive points are within overlapdist
        #   of the initial scatterlen segment.

        # look for transitions from too_close to not too_close, and back
        dclose = too_close[1:] - too_close[:-1]

        # get indices of transitions.
        # ignore first transition away from initial segment
        going_out = np.nonzero(dclose == -1)[0]    # too_close to not too_close
        going_out = going_out[1:]
        going_in = np.nonzero(dclose == 1)[0]      # not too_close to too_close

        # get the lengths of too_close segments
        segment_len_threshold = self.overlapdist_um / BIG_STEP_UM * 2
        for i, in_ind in enumerate(going_in):
            try:
                out_ind = going_out[i]
            except IndexError:
                out_ind = len(dclose)
            if out_ind - in_ind > segment_len_threshold:
                self.overlap = True
                break

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
