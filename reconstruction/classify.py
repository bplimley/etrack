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

        # self.flag_backsteps()
        self.flag_newparticle()
        self.reorder_backsteps(v=verbose)

        self.check_early_scatter()
        self.check_overlap()

    # def flag_backsteps(self):
    #     """
    #     Look for backwards steps (from misordering in DiffuseTrack).
    #     Make an attribute that flags them.
    #
    #     Backwards steps are marked by <0.5um step and >90 degree from previous
    #     step direction.
    #     """
    #
    #     self.dx = self.x[:, 1:] - self.x[:, :-1]
    #     # step lengths
    #     self.d = np.linalg.norm(self.dx, axis=0)
    #     # step directions
    #     self.stepdirs = self.normalize_steps(self.dx, d=self.d)
    #
    #     # cos(difference in direction): dot product.
    #     # (1 = same direction, 0 = 90 degree turn, -1 = opposite direction)
    #     ddir = np.sum(self.stepdirs[:, 1:] * self.stepdirs[:, :-1], axis=0)
    #     # make it the same size as dx.
    #     # ddir[i] is the difference between stepdirs[i-1] and stepdirs[i]
    #     self.ddir = np.concatenate(([0], ddir))
    #
    #     self.dx_backward_flag = (self.d < BIG_STEP_UM / 2) & (self.ddir < 0)

    def flag_newparticle(self):
        """
        Look for particle transitions - jumping to a new electron ID in Geant4.

        Marked by >1.5um step.
        """

        if not hasattr(self, 'd'):
            self.dx = self.x[:, 1:] - self.x[:, :-1]
            self.d = np.linalg.norm(self.dx, axis=0)

        self.dx_newparticle_flag = (self.d > BIG_STEP_UM * 1.5)

    def reorder_backsteps(self, v=False):
        """
        Reorder positions and energies according to dx_backward_flag.
        The before and after positions of the backstep need to be switched.
        """

        self.dx = self.x[:, 1:] - self.x[:, :-1]
        # step lengths
        self.d = np.linalg.norm(self.dx, axis=0)
        # step directions
        self.stepdirs = self.normalize_steps(self.dx, d=self.d)

        # cos(difference in direction): dot product.
        # (1 = same direction, 0 = 90 degree turn, -1 = opposite direction)
        ddir = np.sum(self.stepdirs[:, 1:] * self.stepdirs[:, :-1], axis=0)
        # make it the same size as dx.
        # ddir[i] is the difference between stepdirs[i-1] and stepdirs[i]
        self.ddir = np.concatenate(([0], ddir))
        # print('before: ddir[:50] = ')
        # print(self.ddir[:50])
        self.x0 = np.copy(self.x)
        self.dx0 = np.copy(self.dx)
        self.d0 = np.copy(self.d)
        self.stepdirs0 = np.copy(self.stepdirs)
        self.ddir0 = np.copy(self.ddir)

        #
        strng1 = 'dir0:  '
        strng2 = 'd0:    '
        ddist = directional_distance(self.x)
        for i in xrange(50):
            if ddist[i] > 0:
                strng1 += '+'
            else:
                strng1 += '-'
            if self.d[i] < 0.49:
                strng2 += '.'
            elif self.d[i] < 0.5:
                strng2 += '_'
            elif self.d[i] < 0.99:
                strng2 += '~'
            elif self.d[i] < 1.0:
                strng2 += '-'
            elif self.d[i] < 1.48:
                strng2 += '^'
            elif self.d[i] < 1.5:
                strng2 += '*'
            else:
                strng2 += '#'
        print(strng1)
        print(strng2)

        # how long of a segment are we looking at?
        numsteps = self.scatterlen_um / BIG_STEP_UM * 2

        # only operate on the first particle
        try:
            new_particle_ind = np.nonzero(self.dx_newparticle_flag)[0][0]
        except IndexError:
            # no new particles flagged. just ignore
            new_particle_ind = numsteps + 1
        assert new_particle_ind > numsteps, "Particle track too short"

        i = 1
        self.stepinds = np.arange(numsteps + 1)
        while i < numsteps and i < self.stepdirs.shape[1] - 1:
            ddir = np.sum(self.stepdirs[:, i] * self.stepdirs[:, i + 1],
                          axis=0)
            ansi_reset = '\033[0m'
            ansi_yellow = '\033[33m' + '\033[1m'

            if ddir < -0.7:
                strng = ('{}. dx[i]={}, dx[i+1]={}. d={:.2f}. ' +
                         ansi_yellow + 'ddir={:.2f}' + ansi_reset)
            else:
                strng = '{}. dx[i]={}, dx[i+1]={}. d={:.2f}. ddir={:.2f}'
            if v:
                print(strng.format(
                    i, self.dx[:, i], self.dx[:, i + 1], self.d[i], ddir))
            if ddir < -0.99:
                # backstep. swap entries
                print('Swapping {} and {}'.format(i, i+1))
                (self.x[:, i], self.x[:, i + 1]) = (
                    np.copy(self.x[:, i + 1]),
                    np.copy(self.x[:, i]))
                (self.E[i], self.E[i + 1]) = (
                    np.copy(self.E[i + 1]),
                    np.copy(self.E[i]))
                (self.stepinds[i], self.stepinds[i + 1]) = (
                    np.copy(self.stepinds[i + 1]),
                    np.copy(self.stepinds[i]))

                # update dx and stepdirs
                self.dx[:, i:(i + 2)] = (self.x[:, (i + 1):(i + 3)] -
                                         self.x[:, i:(i + 2)])
                self.d[i:(i + 2)] = np.linalg.norm(
                    self.dx[:, i:(i + 2)], axis=0)
                self.stepdirs[:, i:(i + 2)] = self.normalize_steps(
                    self.dx[:, i:(i + 2)], d=self.d[i:(i + 2)])
            i += 1

        # self.flag_backsteps()
        # self.flag_newparticle()

        # repeat
        self.dx = self.x[:, 1:] - self.x[:, :-1]
        # step lengths
        self.d = np.linalg.norm(self.dx, axis=0)
        # step directions
        self.stepdirs = self.normalize_steps(self.dx, d=self.d)

        # cos(difference in direction): dot product.
        # (1 = same direction, 0 = 90 degree turn, -1 = opposite direction)
        ddir = np.sum(self.stepdirs[:, 1:] * self.stepdirs[:, :-1], axis=0)
        # make it the same size as dx.
        # ddir[i] is the difference between stepdirs[i-1] and stepdirs[i]
        self.ddir = np.concatenate(([0], ddir))

        #
        strng1 = 'dir1:  '
        strng2 = 'd1:    '
        ddist = directional_distance(self.x)
        for i in xrange(50):
            if ddist[i] > 0:
                strng1 += '+'
            else:
                strng1 += '-'
            if self.d[i] < 0.49:
                strng2 += '.'
            elif self.d[i] < 0.5:
                strng2 += '_'
            elif self.d[i] < 0.99:
                strng2 += '~'
            elif self.d[i] < 1.0:
                strng2 += '-'
            elif self.d[i] < 1.48:
                strng2 += '^'
            elif self.d[i] < 1.5:
                strng2 += '*'
            else:
                strng2 += '#'
        print()
        print(strng1)
        print(strng2)

        # print('after: ddir[:50] = ')
        # print(self.ddir[:50])

        self.numsteps = numsteps

    def check_early_scatter(self):
        """
        look for a >90 degree direction change within the first scatterlen_um
        of track.

        track should already be fixed using reorder_backsteps().
        """

        self.early_scatter = np.any(self.ddir[:self.numsteps] < 0)

        verbose = False
        if self.early_scatter and verbose:
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


def directional_distance(x):
    """
    Get the distance of each step, but with a sign corresponding to direction.
    """
    dx = x[:, 1:] - x[:, :-1]
    d = np.linalg.norm(dx, axis=0)
    stepdirs = dx / d
    v0 = stepdirs[:, 0:1]
    dirsign = np.sign(np.sum(stepdirs * v0, axis=0))

    ddist = dirsign * d
    return ddist
