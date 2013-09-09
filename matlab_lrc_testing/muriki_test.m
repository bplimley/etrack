%Define a schedular for torque-pbs
sched = findResource('scheduler','type','torque');

%Set matlab path for the schedular
set(sched, 'ClusterMatlabRoot', '/global/software/centos-5.x86_64/modules/matlab/matlab-R2011b/');

%Set the output path (make a local directory for output and error log)
%set(sched, 'DataLocation', '/global/home/users/kaisong/mdce/data');
set(sched, 'DataLocation', '/global/home/users/bcplimley/multi_angle/test/');

%Set the arguments for qsub (copy your pbs script to here)
set(sched,'SubmitArguments','-q lr_batch -A ac_amsd -l nodes=1:ppn=8:lr -l walltime=00:30:00');

%Create the job from the schedular
j=createJob(sched)

%Create a task: invoke rand(2,5)
%T = createTask(j,@rand,1,{2,5})
T = createTask(j,@MultiAngleDiffuseScript_parallel1,1)

%Submet the job, with tasks in it.
submit(j);

%Wait for the result
unix('echo " "');
unix('echo " "');
unix('echo "waiting for the server to run the job..."');
unix('echo "..."');
waitForState(j);
unix('echo " "');
unix('echo "job is finished."');
result = getAllOutputArguments(j);

%Print out the result
result{1,1}

%hostname
unix('echo " "');
unix('hostname')
unix('echo " "');

