%run HybridTrack_1b on tracks in /mnt/grimdata5/coffer/eT_MultiAngle_Si_Sept2012/Batch01/

% for lawrencium

foldernum = '01';

addpath ./	%not needed
load dEdx_ref.mat

%pixelsize, noise
fieldNames = {'pix2_5noise0', 'pix2_5noise0_003748', ...
    'pix5noise0', 'pix5noise0_01874', ...
    'pix10_5noise0', 'pix10_5noise0_01874', 'pix10_5noise0_0937', 'pix10_5noise0_1874', 'pix10_5noise0_3748', 'pix10_5noise0_937', ...
    'pix20noise0', 'pix20noise0_01874', 'pix20noise0_1874', 'pix20noise0_3748', 'pix20noise0_937', ...
    'pix40noise0', 'pix40noise0_01874', 'pix40noise0_1874', 'pix40noise0_3748', 'pix40noise0_937'};
pixelSize = [2.5*ones(1,2), ...
    5*ones(1,2), ...
    10.5*ones(1,6), ...
    20*ones(1,5), ...
    40*ones(1,5)];
noise = [0, 0.003748, ...
    0, 0.01874, ...
    0, 0.01874, 0.0937, 0.1874, 0.3748, 0.937, ...
    0, 0.01874, 0.1874, 0.3748, 0.937, ...
    0, 0.01874, 0.1874, 0.3748, 0.937];
if ~(length(fieldNames)==length(pixelSize) && length(fieldNames)==length(noise))
    error('variables don''t match size')
end

%GRIM:
%loadpath = '/mnt/grimdata5/coffer/eT_MultiAngle_Si_Sept2012/eT_MultiAngle_Si_Batch01/';
%LRC:
loadpath = ['/global/scratch/bcplimley/multi_angle/newtracks',foldernum,'out/t2/'];
loadpre = 'MultiAngle_Diffused_';
loadsuf = '.mat';

%GRIM:
%savepath = '/mnt/grimdata5/plimley/multi_angle_2012/Batch01/';
%LRC:
savepath = ['/global/scratch/bcplimley/multi_angle/newtracks',foldernum,'_HT/t2/'];
savepre = 'MultiAngle_HT_';
savesuf = '.mat';
placeholdersuf = '_PH.mat';

matlabpool local 10;

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
