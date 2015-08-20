%run HybridTrack_1b on tracks in /mnt/grimdata5/coffer/eT_MultiAngle_Si_Sept2012/Batch01/

% for lawrencium

foldernum = '01';

addpath ./	%not needed
load dEdx_ref.mat

%pixelsize, noise
fieldNames = {'pix2_5noise0', ...
    'pix5noise0', ...
    'pix10_5noise0', ...
    'pix20noise0', ...
    'pix40noise0'};
pixelSize = [2.5, 5, 10.5, 20, 40];
noise = [0, 0, 0, 0, 0];
if ~(length(fieldNames)==length(pixelSize) && length(fieldNames)==length(noise))
    error('variables don''t match size')
end

%GRIM:
%loadpath = '/mnt/grimdata5/coffer/eT_MultiAngle_Si_Sept2012/eT_MultiAngle_Si_Batch01/';
%LRC:
loadpath = ['/global/scratch/bcplimley/multi_angle/newtracks',foldernum,'out/'];
loadpre = 'MultiAngle_Diffused_';
loadsuf = '.mat';

%GRIM:
%savepath = '/mnt/grimdata5/plimley/multi_angle_2012/Batch01/';
%LRC:
savepath = ['/global/scratch/bcplimley/multi_angle/newtracks',foldernum,'_HT/'];
savepre = 'MultiAngle_HT_';
savesuf = '.mat';
placeholdersuf = '_PH.mat';

matlabpool local 6;

flist = dir(fullfile(loadpath,[loadpre,'*',loadsuf]));
disp(['Found ',num2str(length(flist)),' files in ',loadpath,' at ',datestr(now)])

%files in parallel to avoid conflicts
parfor i = 1:length(flist)
    try
        HT_worker(loadpath,flist,savepath,savepre,loadpre,loadsuf,savesuf,noise,pixelSize,fieldNames,i,placeholdersuf,dEdx_ref);
    catch err
        disp(['Error: ',err.message,' on ',flist(i).name])
    end

end
matlabpool close;
