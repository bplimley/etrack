# deviations.py

import numpy as np
import math
import glob
import os
import time

import geant
import tabata

def measure_deviations(electron):
    """
    Measure angular deviation vs. radial distance.

    Input is a geant4 matrix from geant.separateElectrons.
    """

    if electron.shape[0]==0:
        return None, None

    # column indices
    ind_trackID = 0
    ind_parentID = 1
    ind_stepnum = 2
    ind_charge = 3
    ind_initpos = range(4,7)
    ind_finalpos = range(7,10)
    ind_tracklen = 10
    ind_steplen = 11
    ind_final_E = 12
    ind_dE = 13

    energy_keV = geant.measureEnergyKev(electron)
    # tabata_range_um = tabata.extrapolatedRangeSi(energy_keV)

    trackID = electron[:,ind_trackID].astype(int)
    parentID, charge = geant.constructParticleTable(
        electron,ind_trackID,ind_parentID,ind_charge)

    # (copied from geant.measureExtrapolatedRangeX)
    # exclude electrons induced by secondary photons
    #   (e.g. bremsstrahlung)
    #   i.e., only include particles with a pure electron ancestry
    # start from all electrons, and remove any with photon ancestors.
    is_valid = charge==-1
    was_valid = np.ones(len(is_valid))>0
    # is there a better way to make  a boolean array?
    while any(np.logical_xor(is_valid,was_valid)):
        was_valid = is_valid
        is_valid = np.logical_and(
            is_valid,
            np.logical_or(is_valid[parentID], parentID==0))

    is_valid_step = is_valid[trackID]

    first_step = list(electron[is_valid_step,ind_stepnum]).index(1)
    initial_pos = electron[first_step, ind_initpos]
    # assume initial direction is along x-axis

    offset_vector_mm = (electron[is_valid_step,:][:, ind_finalpos] -
                        initial_pos)
    radial_distance_mm = np.sqrt(
        offset_vector_mm[:,0]**2 +
        offset_vector_mm[:,1]**2 +
        offset_vector_mm[:,2]**2)
    atan_y = np.sqrt(
        offset_vector_mm[:,1]**2 + offset_vector_mm[:,2]**2)
    atan_x = offset_vector_mm[:,0]
    deviation_deg = np.arctan2(atan_y, atan_x)

    return radial_distance_mm, deviation_deg

def run_file(filename, savename):
    """
    Load a geant data file, measure deviations, save to file.
    """

    electrons = geant.separateElectrons(geant.loadG4Data(filename))
    energy_keV = [geant.measureEnergyKev(e) for e in electrons]
    radial_distance_mm = [[] for i in xrange(len(electrons))]
    deviation_deg = [[] for i in xrange(len(electrons))]
    for i in xrange(len(electrons)):
        radial_distance_mm[i], deviation_deg[i] = measure_deviations(electrons[i])

    np.savez(savename,
             energy_keV = energy_keV,
             radial_distance_mm = radial_distance_mm,
             deviation_deg = deviation_deg)

def get_file_index(filename):
    """
    E.g. Mat_10009_e_Si_varE_10_cut10um_step10um.dat --> 10009

    Assume there is an underscore before and after the index.
    Assume the index is the first numeric part of the filename.
    """

    substrings = filename.split('_')
    substring_isnumeric = [s.isdigit() for s in substrings]
    index_str = substrings[substring_isnumeric.index(True)]
    return int(index_str)

def pad_file_index(file_index,length):
    """
    E.g. 33 -> '00033'
    """

    return str(file_index).zfill(length)

def construct_filename(filepattern, ind, length):
    """
    E.g. deviations_*_asdf_10um.mat, 33, 5 ->
        deviations_00033_asdf_10um.mat
    """

    asterisk = filepattern.index('*')
    index_str = pad_file_index(ind,length)
    filename = filepattern[:asterisk] + index_str + filepattern[asterisk+1:]
    return filename

def run_directory(filepath, filepattern, savepattern, phpattern, save_subdir='', ph_subdir=''):
    """
    Process data files in a directory, using placeholder files.

    filepath: directory
    filepattern: glob with * in the number section ONLY (i.e. don't abbreviate)
    savepattern: glob with * in the number section
    phpattern: glob with * in the number section
    """

    ind_str_length = 5
    flist = glob.glob(os.path.join(filepath,filepattern))
    # glob output includes path
    flist.sort()

    print "Found", str(len(flist)), "files in",
    print os.path.join(filepath,filepattern), "at", time.ctime()

    for f in flist:
        fname = os.path.basename(f)
        ind = get_file_index(fname)
        ph = construct_filename(phpattern, ind, ind_str_length)
        phfull = os.path.join(filepath,ph_subdir,ph)
        savename = construct_filename(savepattern, ind, ind_str_length)
        savefull = os.path.join(filepath,save_subdir,savename)
        if os.path.isfile(savefull):
            continue
        if os.path.isfile(phfull):
            continue
        # double check... try to avoid threads duplicating work
        time.sleep(2*np.random.random())
        if os.path.isfile(phfull):
            continue
        print "Beginning", f, "at", time.ctime()
        # create placeholder
        with open(phfull,'w') as phid:
            pass
        # operate on file
        run_file(f,savefull)
        # remove placeholder
        os.remove(phfull)
        print "Finished", f, "at", time.ctime()
