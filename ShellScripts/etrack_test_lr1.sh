#PBS -q lr_batch
#PBS -A ac_amsd
#PBS -N etrack_test_lr1
#PBS -l nodes=1:ppn=8:lr1
#PBS -l walltime=24:00:00
#PBS -o /global/scratch/bcplimley/multi_angle/G4TH_out_lr1
#PBS -e /global/scratch/bcplimley/multi_angle/G4TH_err_lr1
#PBS -M brianp@berkeley.edu
#PBS -m bea

/bin/bash
module load matlab/R2011b

cd /global/scratch/bcplimley/multi_angle

set MNAME=MultiAngleDiffuseScript_parallel2_lr1.m

unset DISPLAY

matlab -nodisplay -nodesktop -nosplash -r $MNAME

