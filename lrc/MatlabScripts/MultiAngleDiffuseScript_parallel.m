%run Geant4TrackHandling7 on tracks in /mnt/grimdata5/coffer/eT_MultiAngle_Si_Sept2012/Batch01/

%modified for lawrencium

foldernum = '01';

addpath ./	%not needed
addpath /global/home/users/bcplimley/improc/
addpath /global/home/users/bcplimley/cfit/

pixelSize = [2.5,5,10.5,20,40];
nb = 0.01874;	%base noise level
noise = nb.*[0,0.2,0.5,1,2,5,10,20,50];

pixelNoiseMatrix = true(length(pixelSize),length(noise));
pixelNoiseMatrix(1,4:7) = false;
pixelNoiseMatrix(2,7) = false;
pixelNoiseMatrix(4:5,2:3) = false;

%GRIM:
%loadpath = '/mnt/grimdata5/coffer/eT_MultiAngle_Si_Sept2012/eT_MultiAngle_Si_Batch01/';
%LRC:
loadpath = ['/global/scratch/bcplimley/multi_angle/newtracks',foldernum,'/'];
loadpre = 'Mat_MultiAngle_CCD_eTracks_662_500k_';
loadsuf = '_TRK.mat';

%GRIM:
%savepath = '/mnt/grimdata5/plimley/multi_angle_2012/Batch01/';
%LRC:
savepath = ['/global/scratch/bcplimley/multi_angle/newtracks',foldernum,'out/'];
savepre = 'MultiAngle_Diffused_';
savesuf = '.mat';

matlabpool local 12;

flist = dir(fullfile(loadpath,[loadpre,'*',loadsuf]));
disp(['Found ',num2str(length(flist)),' files in ',loadpath,' at ',datestr(now)])

%files in parallel to avoid conflicts
parfor i = 1:length(flist)
    MultiAngle_worker(loadpath,flist,savepath,savepre,loadpre,loadsuf,savesuf,noise,pixelSize,pixelNoiseMatrix,i);

end
matlabpool close;
