#PBS -q lr_batch
#PBS -A ac_amsd
#PBS -N HT_3_3
#PBS -l nodes=1:ppn=12:lr2
#PBS -l walltime=12:00:00
#PBS -o /global/scratch/bcplimley/multi_angle/HT_out3_3
#PBS -e /global/scratch/bcplimley/multi_angle/HT_err3_3
#PBS -M brianp@berkeley.edu
#PBS -m be

/bin/bash
module load matlab/R2011b

cd /global/scratch/bcplimley/multi_angle


unset DISPLAY

matlab -nodisplay -nosplash -nodesktop -r HT_parallel1_3

