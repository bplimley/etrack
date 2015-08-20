#PBS -q lr_batch
#PBS -A ac_amsd
#PBS -N etrack_24h_8cpu_8
#PBS -l nodes=1:ppn=12:lr2
#PBS -l walltime=24:00:00
#PBS -o /global/scratch/bcplimley/multi_angle/G4TH_out8
#PBS -e /global/scratch/bcplimley/multi_angle/G4TH_err8
#PBS -M brianp@berkeley.edu
#PBS -m bea

/bin/bash
module load matlab/R2011b

cd /global/scratch/bcplimley/multi_angle


unset DISPLAY

matlab -nodisplay -nosplash -nodesktop -r MultiAngleDiffuseScript_parallel1 

