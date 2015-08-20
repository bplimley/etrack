#PBS -q lr_batch
#PBS -A ac_amsd
#PBS -N etrack_test
#PBS -l nodes=1:ppn=12:lr2
#PBS -l walltime=3:00:00
#PBS -o /global/scratch/bcplimley/multi_angle/G4TH_out1b
#PBS -e /global/scratch/bcplimley/multi_angle/G4TH_err1b
#PBS -M brianp@berkeley.edu
#PBS -m bea

/bin/bash
module load matlab/R2011b

cd /global/scratch/bcplimley/multi_angle


unset DISPLAY

matlab -nodisplay -nosplash -nodesktop -r MultiAngleDiffuseScript_parallel1 

