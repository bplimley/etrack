#PBS -q lr_batch
#PBS -A ac_amsd
#PBS -N eserial_2h_1cpu_13
#PBS -l nodes=1:ppn=2:lr2
#PBS -l walltime=2:00:00
#PBS -o /global/scratch/bcplimley/multi_angle/G4TH_out13
#PBS -e /global/scratch/bcplimley/multi_angle/G4TH_err13
#PBS -M brianp@berkeley.edu
#PBS -m e

/bin/bash
module load matlab/R2011b

cd /global/scratch/bcplimley/multi_angle


unset DISPLAY

matlab -nodisplay -nosplash -nodesktop -r MultiAngleDiffuseScript_serial1 > outserial13_1 
