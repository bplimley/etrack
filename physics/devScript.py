# devScript.py
#
# copied from rangeScript.py 3/6/2015

import numpy as np
import math
import os
import glob
import time
import datetime
import sys
sys.path.reverse()
sys.path.append('/global/home/users/bcplimley/repo/plimley/python/etrack/physics/')
sys.path.reverse()

import geant
import deviations

def mainscript():
    baseDir = '/global/scratch/bcplimley/electrons/parameterTests/'
    dirform = 'e_Si*'
    fform = 'Mat_*.dat'
    
    batchSize = 10
    
    dirlist = glob.glob(os.path.join(baseDir,dirform))
    dirlist = [os.path.split(d)[-1] for d in dirlist]
    
    print('Found ' + str(len(dirlist)) + ' directories')
    print('')
    
    # stagger the start time, for multithreading ease
    # everything else is sequential with placeholders
    timeLength = np.random.random(1)[0]*300
    time.sleep(timeLength)
    
    for d in dirlist:
        # construct filepattern with * only replacing the index
        filepattern = 'Mat_*_' + d + '.dat'
        phpattern = 'ph_*_' + d + '.ph'
        savepattern = 'dev_*_' + d # auto npz extension
        
        dirfull = os.path.join(baseDir,d)
        deviations.run_directory(dirfull,filepattern,
            savepattern,phpattern)

mainscript()

