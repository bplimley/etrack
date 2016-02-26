# geant.py
#

import numpy as np

nan = float('nan')


def loadG4Data(filename):
    """Read Geant4 data from *.dat data file.

    The data consists of comma-separated values, ~15 columns, and
    some thousands or more rows, amounting to perhaps ~400 MB. In
    MATLAB I operate on it as a single 2D array/matrix and for now
    I have maintained that here for now.
    """

    f = open(filename, 'r')
    n = 1
    data = []
    rows = f.readlines()
    for r in rows:
        data.append([float(num) for num in r.rstrip().split(',')])
    f.close()

    return np.array(data)    # numpy


def saveG4Data(matrix, filename):
    """Write G4 data to file in NumPy format.

    (I considered HDF5 via PyTables, but that would require a little
    more work.)
    """

    f = open(filename, 'w')
    np.save(f, np.array(matrix))
    f.close()

    return 0


def separateElectrons(matrix):
    """Break a G4 data matrix apart into individual electrons.

    The data matrix represents the trajectories and interactions of a
    number of separate energetic electrons. Each electron's
    interactions are simulated in Monte Carlo fashion by sampling
    physics-based probability distributions for each step of its
    trajectory. Anyway, each electron is used separately for the
    statistical analyses, but for the Geant4 simulation and file
    management it is convenient to group them together in batches
    within each file.

    Each row of the matrix is one step of one electron.

    The first row of each separate primary electron can be identified
    by parentID==0 (i.e. the particle making this step was not produced
    by the interactions of any other particle) and stepnum==1 (this is
    the first step of this particle).

    These are the 'primary' electrons, but as they travel through the
    material and collide with atoms, there are also 'secondary'
    electrons generated, which have a different particle ID, but are
    always analyzed together with the primary electron. Thus, my
    'electron' array is actually the data corresponding to one primary
    electron and all secondary (tertiary, etc.) electrons produced by
    it.
    """

    # careful, columns are indexed from 0!
    # column "1" = parentID; column "2" = step num

    isInitialStep = np.logical_and(matrix[:, 1] == 0, matrix[:, 2] == 1)
    initialStep = np.array(np.nonzero(isInitialStep), dtype=int)[0, :]

    finalStep = np.zeros(initialStep.shape, dtype=int)
    finalStep[:-1] = [initialStep[i+1]-1 for i in range(len(initialStep)-1)]
    finalStep[-1] = matrix.shape[0] - 1

    electron = [
        np.array(matrix[initialStep[i]:finalStep[i], :])
        for i in range(len(initialStep))]

    return electron


def measureEnergyKev(electron):
    """Measure the initial energy of an electron.

    I generate electrons with different initial energies, and the
    initial energy can be identified by summing two columns in the
    first step of the primary electron.
    """

    indFinalE = 12
    inddE = 13

    if electron.shape[0] == 0:
        # nothing recorded! range cut too big
        return 0.0

    energyEv = electron[0, indFinalE] + electron[0, inddE]
    energyKev = energyEv / 1000.0

    return energyKev


def measureExtrapolatedRangeX(electron):
    """Measure the range in um along the x-direction of an electron.

    Secondary electrons induced by bremsstrahlung are excluded.

    The farthest extent of the electron trajectory along its initial
    direction of travel is a useful benchmark to confirm that Geant4
    is doing physics properly with the Geant4 settings I have chosen.

    Every electron in these simulations starts out traveling along
    (1,0,0) and that simplifies the measurement.
    """

    # column indices
    #   offset by 1 according to 0-base vs. MATLAB's 1-base.
    indTrackID = 0
    indParentID = 1
    indStepNum = 2
    indCharge = 3
    indInitPos = range(4, 7)
    indFinalPos = range(7, 10)
    indTrackLen = 10
    indStepLen = 11
    indFinalE = 12
    inddE = 13

    if electron.shape[0] == 0:
        # nothing recorded! range cut too big
        return 0.0

    trackID = electron[:, indTrackID].astype(int)
    parentID, charge = constructParticleTable(
        electron, indTrackID, indParentID, indCharge)
    # parentID and charge are 1-indexed
    #   (i.e., charge[0] doesn't mean anything)

    # exclude electrons induced by secondary photons (e.g. bremsstrahlung)
    #   i.e., only include particles with a pure electron ancestry
    # start from all electrons, and remove any with photon ancestors.
    isValid = charge == -1
    wasValid = np.ones(len(isValid)) > 0    # better way to make boolean array?
    while any(np.logical_xor(isValid, wasValid)):
        wasValid = isValid
        isValid = np.logical_and(
            isValid,
            np.logical_or(isValid[parentID], parentID == 0))

    firstStep = list(electron[:, indStepNum]).index(1)
    x0 = electron[firstStep, indInitPos[0]]

    isValidStep = isValid[trackID]
    rangeMm = max(electron[isValidStep, indInitPos[0]] - x0)
    rangeUm = rangeMm * 1000

    return rangeUm


def constructParticleTable(electron, indTrackID, indParentID, indCharge):
    """Return list of parentID, charge for all particles.

    For the range, it is important to understand where the secondary
    particles have come from, spefically whether the primary electron
    generated a secondary photon (bremsstrahlung process) which then
    generated more electrons.

    The particle table lists the parent ID and electric charge for each
    particle ID.
    """

    if electron.shape[0] == 0:
        return None, None

    # N = int(max(electron[:,indTrackID]))
    particleList = np.unique(electron[:, indTrackID]).astype(int)

    # parentID and charge are initialized with zeros.
    # This should be okay because charge==0 means not an electron,
    #   so it is essentially ignored in measureExtrapolatedRangeX.
    # The +1 is so that parentID and charge can be indexed by
    #   TrackID, which is starts from 1 not 0.
    parentID = np.zeros(max(particleList)+1)
    charge = np.zeros(max(particleList)+1)

    for i in particleList:
        thisStartIndex = list(electron[:, indTrackID]).index(i)
        parentID[i] = electron[thisStartIndex, indParentID]
        charge[i] = electron[thisStartIndex, indCharge]

    # if any(isnan(parentID)) or any(isnan(charge)),
    #   this does NOT generate an error but produces the most negative
    #   int64 value.
    return parentID.astype(int), charge.astype(int)


class G4Electron():
    """
    Geant4 electron object.
    """

    indTrackID = 0
    indParentID = 1
    indStepNum = 2
    indCharge = 3
    indInitPos = range(4, 7)
    indFinalPos = range(7, 10)
    indTrackLen = 10
    indStepLen = 11
    indFinalE = 12
    inddE = 13

    def __init__(self, G4matrix=None):
        if G4matrix is not None:
            self.set_matrix(G4matrix)

    def set_matrix(self, G4matrix):
        """Set G4 matrix, updating energy and particle table."""

        self.__mat = G4matrix
        self.set_energy()
        self.construct_particle_table()

    def get_matrix(self):
        return self.__mat.c

    def set_energy(self):
        self.__energy_kev = measureEnergyKev(self.get_matrix)

    def get_energy(self):
        try:
            return self.__energy_kev
        except something:
            return None

    def construct_particle_table(self):
        self.__particle_table = (
            constructParticleTable(self.get_matrix, 0, 1, 3))

    def get_particle_table(self):
        return self.__particle_table
