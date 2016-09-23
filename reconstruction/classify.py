# classify a g4track

import numpy as np
# import matplotlib.pyplot as plt

from etrack.reconstruction.trackdata import G4Track
import etrack.io.dataformats as df
import etrack.io.trackio as trackio
# import etrack.reconstruction.evaluation as ev
import etrack.visualization.trackplot as tp

BIG_STEP_UM = 1.0   # micron


class Classifier(object):
    """
    Object to handle auto classifying (ground truth) a Monte Carlo track
    """

    class_name = 'Classifier'
    data_format = df.get_format(class_name)

    def __init__(self, g4track, suppress_check=False):
        assert isinstance(g4track, G4Track), "Not a G4Track object"
        assert g4track.x.shape[0] == 3, "x has funny shape"
        self.x = np.copy(g4track.x)
        self.E = np.copy(g4track.dE.flatten())
        self.g4track = g4track

        # avoid IO errors in case of only mc_classify and not end_classify
        # or for errors in test_moments.classifierlist_from_tracklist

        self.scatterlen_um = None
        self.overlapdist_um = None
        self.scatter_type = None
        self.use2d_angle = None
        self.use2d_dist = None
        self.angle_threshold_deg = None
        self.escaped = None
        self.early_scatter = None
        self.total_scatter_angle = None
        self.overlap = None

        self.wrong_end = None
        self.n_ends = None
        self.max_end_energy = None
        self.min_end_energy = None

        self.error = None

    @classmethod
    def from_hdf5(cls, h5group, h5_to_pydict=None, pydict_to_pyobj=None,
                  reconstruct=False):
        """
        Initialize a Classifier object from an HDF5 group.
        """

        if h5_to_pydict is None:
            h5_to_pydict = {}
        if pydict_to_pyobj is None:
            pydict_to_pyobj = {}

        read_dict = trackio.read_object_from_hdf5(
            h5group, h5_to_pydict=h5_to_pydict)

        constructed_object = cls.from_pydict(
            read_dict, pydict_to_pyobj=pydict_to_pyobj,
            reconstruct=reconstruct)

        return constructed_object

    @classmethod
    def from_pydict(cls, read_dict, pydict_to_pyobj=None, reconstruct=False):
        """
        Initialize a Classifier object from a pydict.
        """

        if pydict_to_pyobj is None:
            pydict_to_pyobj = {}

        if id(read_dict) in pydict_to_pyobj:
            return pydict_to_pyobj[id(read_dict)]

        # first, reconstruct g4track (if needed)
        if isinstance(read_dict['g4track'], G4Track):
            # g4track is already a G4Track object (not sure how)
            g4track = read_dict['g4track']
        elif (isinstance(read_dict['g4track'], dict) and
                id(read_dict['g4track']) in pydict_to_pyobj):
            # g4track is in the pydict table
            g4track = pydict_to_pyobj[id(read_dict['g4track'])]
        elif isinstance(read_dict['g4track'], dict):
            # g4track not in the pydict table. create and add it
            g4track = G4Track.from_pydict(
                read_dict['g4track'], pydict_to_pyobj=pydict_to_pyobj)
            pydict_to_pyobj[id(read_dict['g4track'])] = g4track
        else:
            raise Exception("Unexpected or missing 'g4track' in Classifier")

        new_obj = cls(g4track)

        # add entry to pydict_to_pyobj
        pydict_to_pyobj[id(read_dict)] = new_obj

        # fill in outputs
        if read_dict['error'] is not None:
            new_obj.error = read_dict['error']
        if read_dict['scatterlen_um'] is not None:
            # mc_classify ran
            new_obj.scatterlen_um = read_dict['scatterlen_um']
            new_obj.overlapdist_um = read_dict['overlapdist_um']
            new_obj.escaped = read_dict['escaped']
            new_obj.scatter_type = read_dict['scatter_type']
            new_obj.use2d_angle = read_dict['use2d_angle']
            new_obj.use2d_dist = read_dict['use2d_dist']
            new_obj.angle_threshold_deg = read_dict['angle_threshold_deg']
        if read_dict['early_scatter'] is not None:
            # no TrackTooShortError
            new_obj.early_scatter = read_dict['early_scatter']
            new_obj.total_scatter_angle = read_dict['total_scatter_angle']
            new_obj.overlap = read_dict['overlap']
        if read_dict['wrong_end'] is not None:
            # end_classify ran
            new_obj.wrong_end = read_dict['wrong_end']
            new_obj.n_ends = read_dict['n_ends']
            new_obj.max_end_energy = read_dict['max_end_energy']
            new_obj.min_end_energy = read_dict['min_end_energy']

        if reconstruct:
            new_obj.mc_classify()
            print(
                'Reconstructing Classifier.end_classify() not implemented yet')

        return new_obj

    def mc_classify(self, scatterlen_um=25, overlapdist_um=40, verbose=False):
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

        self.check_escape()

        self.flag_newparticle()

        self.check_early_scatter(v=verbose)
        self.check_overlap()

    def end_classify(self, track, mom=None, HT=None):
        """
        Look at the end segment selection algorithm results, and classify.
        """

        self.get_end_info(track, mom=mom, HT=HT)
        self.check_wrong_end(track, mom=mom, HT=HT)

    def get_end_info(self, track, mom=None, HT=None):
        """
        Record number of ends, minimum end energy (this is the chosen end),
        maximum end energy (may indicate escape).
        """

        if mom is not None:
            ends_energy = mom.info.ends_energy
        elif HT is not None:
            ends_energy = HT.info.ends_energy

        self.n_ends = len(ends_energy)
        self.max_end_energy = np.max(ends_energy)
        self.min_end_energy = np.min(ends_energy)

    def check_wrong_end(self, track, mom=None, HT=None, maxdist=3):
        """
        See if the algorithm's end segment was correct or not.
        """

        if mom is None and HT is None:
            raise ValueError(
                'Requires either a moments object or a HybridTrack object')

        g4xfull, g4yfull = tp.get_image_xy(track)
        g4x, g4y = g4xfull[0], g4yfull[0]

        # could throw an AttributeError if there were errors in the algorithm
        if mom:
            algx, algy = mom.start_coordinates
        elif HT:
            algx, algy = HT.start_coordinates
        else:
            raise ValueError('bad value in moments or HybridTrack object')

        dist = np.sqrt((algx - g4x)**2 + (algy - g4y)**2)
        self.end_distance = dist
        if dist > maxdist:
            self.wrong_end = True
        else:
            self.wrong_end = False

        self.g4xy = g4x, g4y
        self.algxy = algx, algy

    def check_escape(self):
        """
        Check the Etot and Edep to see if track escaped. (>2 keV difference)
        Unfortunately the g4track.is_contained flag is None in this dataset.
        """

        energy_diff_kev = (
            self.g4track.energy_tot_kev - self.g4track.energy_dep_kev)
        self.escaped = (energy_diff_kev > 2.0)

    def flag_newparticle(self):
        """
        Look for particle transitions - jumping to a new electron ID in Geant4.

        Marked by >1.5um step.
        """

        if not hasattr(self, 'd'):
            self.dx = self.x[:, 1:] - self.x[:, :-1]
            self.d = np.linalg.norm(self.dx, axis=0)

        self.dx_newparticle_flag = (self.d > BIG_STEP_UM * 1.5)

    def check_early_scatter(self, v=False, scatter_type='total',
                            use2d_angle=True, use2d_dist=True,
                            angle_threshold_deg=30):
        """
        look for a >90 degree direction change within the first scatterlen_um
        of track.

        v: verbosity (True: print "Early scatter!")
        scatter_type:
          'total': compare direction at end of segment, to beginning of segment
          'discrete': look for a single scattering of more than angle
        angle_threshold_deg: threshold angle. scattering through more than
          this angle is flagged.
        use2d_angle: flag for looking at the scatter angle in the 2D plane
        use2d_dist: flag for measuring scatterlen along the track projection
        """

        angle_threshold_rad = np.float(angle_threshold_deg) / 180 * np.pi
        angle_threshold_cos = np.cos(angle_threshold_rad)

        self.scatter_type = scatter_type
        self.use2d_angle = use2d_angle
        self.use2d_dist = use2d_dist
        self.angle_threshold_deg = angle_threshold_deg

        # use only every other point, so as to bypass the zigzag issue.
        # x2: positions (every other point)
        self.x2 = self.x[:, ::2]
        # dx2: delta-position between each entry in x2
        self.dx2 = self.x2[:, 1:] - self.x2[:, :-1]

        # integrate the path length, either in 2D or 3D, to determine cutoff
        #   for scatterlen
        # d2: dx2 integrated to get path length
        # d2_2d: dx2 integrated to get path length, in 2D
        if use2d_dist:
            self.d2_2d = np.linalg.norm(self.dx2[:2, :], axis=0)
            integrated_dist = np.cumsum(self.d2_2d)
        else:
            self.d2 = np.linalg.norm(self.dx2, axis=0)
            integrated_dist = np.cumsum(self.d2)
        # ind2: the index where path length (2D or 3D) exceeds scatterlen
        # short tracks raise an IndexError here
        try:
            ind2 = np.nonzero(integrated_dist >= self.scatterlen_um)[0][0] - 1
        except IndexError:
            raise TrackTooShortError()

        # trim x2 and dx2
        self.x2 = self.x2[:, :ind2]
        self.dx2 = self.dx2[:, :ind2]

        # dx2norm: unit vectors of dx2
        self.dx2norm = self.normalize_steps(self.dx2)
        # ddir2: dot product of consecutive dx2norm's. =cos(theta)
        self.ddir2 = np.sum(
            self.dx2norm[:, 1:] * self.dx2norm[:, :-1], axis=0)

        # dx2norm_2d: 2D unit vectors of dx2
        self.dx2norm_2d = self.dx2[:2, :] / np.linalg.norm(
            self.dx2[:2, :], axis=0)
        # ddir2_2d: dot product of consecutive dx2norm_2d's. =cos(theta)
        self.ddir2_2d = np.sum(
            self.dx2norm_2d[:, 1:] * self.dx2norm_2d[:, :-1], axis=0)

        # for discrete scatters, look for large angles in dx2norm or dx2norm_2d
        # for total scatter angle, dot the unit vector with the initial
        #   direction
        if scatter_type.lower() == 'total':
            # ddir: the unit vector at each step,
            #   dotted with the initial unit vector, to get angle of deviation
            # take the maximum from along scatterlen
            if use2d_angle:
                ddir = np.array([
                    np.sum(self.dx2norm_2d[:, 0] * self.dx2norm_2d[:, i])
                    for i in xrange(1, self.dx2norm_2d.shape[1])])
            else:
                ddir = np.array([
                    np.sum(self.dx2norm[:, 0] * self.dx2norm[:, i])
                    for i in xrange(1, self.dx2norm.shape[1])])
            self.early_scatter = (np.min(ddir) < angle_threshold_cos)
            self.total_scatter_angle = np.arccos(np.min(ddir))
        elif scatter_type.lower() == 'discrete':
            if use2d_angle:
                self.early_scatter = np.any(self.ddir2 < angle_threshold_cos)
            else:
                self.early_scatter = np.any(self.ddir2 < angle_threshold_cos)
        else:
            raise ValueError(
                'scatter_type {} not recognized!'.format(scatter_type))

        if self.early_scatter and v:
            print('Early scatter!')

    def check_overlap(self):
        """
        look for a section of track overlapping with the initial scatterlen_um
        of track.
        """

        self.overlap = False    # until shown otherwise
        self.scatterlen_steps = self.scatterlen_um / BIG_STEP_UM * 2

        # see which points are within overlapdist_um of these points
        # distance matrix: distance of all points from each of the first 50

        # arrays of dimensions (self.x.shape[1], self.numsteps)
        all_x, init_x = np.meshgrid(
            self.x[0, :self.scatterlen_steps], self.x[0, :])
        all_y, init_y = np.meshgrid(
            self.x[1, :self.scatterlen_steps], self.x[1, :])

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


class TrackTooShortError(Exception):
    pass


def test_io(tracks=None):
    """
    """

    import etrack.reconstruction.test_moments as tm

    if tracks is None:
        tracks = tm.get_tracklist(n_files=1)
    mom = tm.momentlist_from_tracklist(tracks, fill_nans=False)
    cl = tm.classifierlist_from_tracklist(tracks, mom)

    filename = 'testcl.h5'
    print('Writing classifiers to {}...'.format(filename))
    trackio.write_object_list_to_hdf5(filename, cl, prefix='cl_')

    print('Reading classifiers back...')
    clread = trackio.read_object_list_from_hdf5(
        filename, Classifier.from_hdf5, prefix='cl_')

    print('Checking attributes...')
    attrs = df.get_format('Classifier')
    error_ind = []
    error_list = []
    for i, c in enumerate(clread):
        if hasattr(c, 'error'):
            if c.error:
                error_ind.append(i)
                error_list.append(c.error)
                # if there's a track error, which fields are filled, are wonky
                continue
        for attr in attrs:
            if attr.name == 'g4track':
                continue
            elif getattr(cl[i], attr.name) is not None:
                assert getattr(cl[i], attr.name) == getattr(c, attr.name)

    return error_ind, error_list
