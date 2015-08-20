#PBS -q lr_batch
#PBS -A ac_amsd
#PBS -N etrack_6h_8cpu_18
#PBS -l nodes=1:ppn=12:lr2
#PBS -l walltime=6:00:00
#PBS -o /global/scratch/bcplimley/multi_angle/G4TH_out18
#PBS -e /global/scratch/bcplimley/multi_angle/G4TH_err18
#PBS -M brianp@berkeley.edu
#PBS -m e

/bin/bash
module load matlab/R2011b

cd /global/scratch/bcplimley/multi_angle


unset DISPLAY

matlab -nodisplay -nosplash -nodesktop -r MultiAngleDiffuseScript_parallel1 

