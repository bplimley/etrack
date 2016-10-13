% DT_to_hdf5.m
% copied from MultiAngle_to_hdf5.m

LRCflag = false;
if LRCflag
    % run on LRC
    cd('/global/home/users/bcplimley/multi_angle/MatlabScripts')
    loadpath = '/global/home/users/bcplimley/multi_angle/DTbatch01';
    savepath = '/global/home/users/bcplimley/multi_angle/DTbatch01_h5';
    addpath('../MatlabFunctions')
else
    % LBL desktop testing
    loadpath = '/media/plimley/TEAM 7B/DTbatch01';
    savepath = '/media/plimley/TEAM 7B/DTbatch01_h5';
end

filepattern = 'MultiAngle*.mat';

% edit here for running in parallel
% index = 1;

flist = dir(fullfile(loadpath,filepattern));

% firstind = 1;
% lastind = length(flist);
% numWorkers = 10;
%
% allind = linspace(firstind, lastind, numWorkers+1);
% startind = round(allind(1:end-1));
% endind = round(allind(2:end));  % close enough

if LRCflag
    matlabpool local 10
else
    matlabpool local 4
end

parfor i = 1:length(flist)
    loadname = flist(i).name;
    savename = [loadname(1:end-4),'.h5'];
    donename = ['done_', loadname];

    loadfull = fullfile(loadpath, loadname);
    savefull = fullfile(savepath, savename);
    donefull = fullfile(savepath, donename);

    if isempty(dir(donefull))
        disp(['Beginning ',flist(i).name,' at ',datestr(now)])
        try
            write_DT_hdf5(loadfull, savefull, donefull);
        catch err
            disp(['Error in ',flist(i).name,': ',err.message])
        end
        disp(['-- Finished ',flist(i).name,' at ',datestr(now)])
    end
end

matlabpool close
