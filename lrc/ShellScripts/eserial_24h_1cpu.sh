#PBS -q lr_batch
#PBS -A ac_amsd
#PBS -N eserial_24h_1cpu_14
#PBS -l nodes=1:ppn=2:lr2
#PBS -l walltime=24:00:00
#PBS -o /global/scratch/bcplimley/multi_angle/G4TH_out14
#PBS -e /global/scratch/bcplimley/multi_angle/G4TH_err14
#PBS -M brianp@berkeley.edu
#PBS -m e

/bin/bash
module load matlab/R2011b

cd /global/scratch/bcplimley/multi_angle


unset DISPLAY

matlab -nodisplay -nosplash -nodesktop -r MultiAngleDiffuseScript_serial1 > outserial14
