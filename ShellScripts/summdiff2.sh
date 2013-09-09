#PBS -q lr_batch
#PBS -A ac_amsd
#PBS -N sd_2
#PBS -l nodes=1:ppn=4:lr2
#PBS -l walltime=6:00:00
#PBS -o /global/scratch/bcplimley/multi_angle/SD_out2
#PBS -e /global/scratch/bcplimley/multi_angle/SD_err2
#PBS -M brianp@berkeley.edu
#PBS -m e

/bin/bash
module load matlab/R2011b

cd /global/scratch/bcplimley/multi_angle


unset DISPLAY

matlab -nodisplay -nosplash -nodesktop -r summarizeDiffusion2

