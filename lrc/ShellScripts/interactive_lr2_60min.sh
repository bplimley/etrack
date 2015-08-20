#PBS -q lr_batch
#PBS -A ac_amsd
#PBS -N etrack_interactive_session
#PBS -l nodes=1:ppn=12:lr2
#PBS -l walltime=0:60:00
#PBS -o /global/scratch/bcplimley/multi_angle/interactive_out
#PBS -e /global/scratch/bcplimley/multi_angle/interactive_err
#PBS -M brianp@berkeley.edu
#PBS -m bea

/bin/bash
module load matlab/R2011b

cd /global/scratch/bcplimley/multi_angle


unset DISPLAY

#matlab -nodisplay -nosplash -nodesktop -r HTscript_master1 

