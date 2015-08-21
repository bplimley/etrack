# rangeScript.py

import numpy as np
import math
import os
import glob

import geant as geant

baseDir = '/global/scratch/bcplimley/electrons/parameterTests/'
dirform = 'e_Si_*'
fform = 'Mat_*.dat'

dirlist = glob.glob(os.path.join(baseDir,dirform))

print('Found ' + str(len(dirlist)) + 'directories')
print('')

for d in dirlist[0]:
	flist = glob.glob(os.path.join(baseDir,d,fform))
	fullflist = os.dirlist(os.path.join(baseDir,d))
	print('Found ' + str(len(flist)) + ' files in ' + d)
	print('')

	for f in flist:
		savename = 'ranges_' + f[4:-4] + '.txt'
		if savename in fullflist:
			continue
		electron = geant.separateElectrons(geant.loadG4Data(
			os.path.join(baseDir,d,f)))
		savefile = open(os.path.join(baseDir,d,savename),'w')
		for e in electron:
			energyString = str(geant.measureEnergyKev(e))
			rangeString = str(geant.measureExtrapolatedRangeX(e))
			writeString = energyString + ',' + rangeString + '\n'
			savefile.write(writeString)
		savefile.close
		print('Finished ' + f)
