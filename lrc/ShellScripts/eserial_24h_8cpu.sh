#PBS -q lr_batch
#PBS -A ac_amsd
#PBS -N eserial_24h_8cpu_9
#PBS -l nodes=1:ppn=12:lr2
#PBS -l walltime=24:00:00
#PBS -o /global/scratch/bcplimley/multi_angle/G4TH_out9
#PBS -e /global/scratch/bcplimley/multi_angle/G4TH_err9
#PBS -M brianp@berkeley.edu
#PBS -m bea

/bin/bash
module load matlab/R2011b

cd /global/scratch/bcplimley/multi_angle


unset DISPLAY

matlab -nodisplay -nosplash -nodesktop -r MultiAngleDiffuseScript_serial1 > outserial9_1 &
matlab -nodisplay -nosplash -nodesktop -r MultiAngleDiffuseScript_serial1 > outserial9_2 &
matlab -nodisplay -nosplash -nodesktop -r MultiAngleDiffuseScript_serial1 > outserial9_3 &
matlab -nodisplay -nosplash -nodesktop -r MultiAngleDiffuseScript_serial1 > outserial9_4 &

matlab -nodisplay -nosplash -nodesktop -r MultiAngleDiffuseScript_serial1 > outserial9_5 &
matlab -nodisplay -nosplash -nodesktop -r MultiAngleDiffuseScript_serial1 > outserial9_6 &
matlab -nodisplay -nosplash -nodesktop -r MultiAngleDiffuseScript_serial1 > outserial9_7 &
matlab -nodisplay -nosplash -nodesktop -r MultiAngleDiffuseScript_serial1 > outserial9_8 &
