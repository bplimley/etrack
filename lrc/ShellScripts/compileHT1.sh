#PBS -q lr_batch
#PBS -A ac_amsd
#PBS -N compileHT_1
#PBS -l nodes=1:ppn=4:lr2
#PBS -l walltime=6:00:00
#PBS -o /global/scratch/bcplimley/multi_angle/compile1_out
#PBS -e /global/scratch/bcplimley/multi_angle/compile1_err
#PBS -M brianp@berkeley.edu
#PBS -m be

/bin/bash
module load matlab/R2011b

cd /global/scratch/bcplimley/multi_angle


unset DISPLAY

matlab -nodisplay -nosplash -nodesktop -r HTcompile1

