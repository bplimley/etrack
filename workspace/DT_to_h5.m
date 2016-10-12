function DT_to_h5
    % load a DT .mat file
    % save to a DT .h5 file, readable by python
    % Oct. 12, 2016
    
    % abandoned in favor of DT_to_hdf5.m and write_DT_hdf5.m

    loadpath = '/global/home/users/bcplimley/multi_angle/DTbatch01';
    loadpattern = 'MultiAngle_DT_*.mat';

    savepath = '/global/home/users/bcplimley/multi_angle/DTbatch01_h5';

    flist = dir(fullfile(loadpath, loadpattern));

    for i = 1:length(flist)
        loadname = fullfile(loadpath, flist(i).name)
        savename = fullfile(savepath, flist(i).name)
        do_one_file(loadname, savename)
    end
end

function do_one_file(loadname, savename)

    disp(['Starting ', savename, ' at ', datestr(now)])

    f = load(loadname);
    for k = 1:length(f.DT)
        evtname = ['/', sprintf('%05u', k-1)];
        do_one_event(f.DT{k}, savename, evtname)
    end
end

function do_one_event(evt, savename, evtname)

    if isempty(evt)
        err = 'DT cell was empty'
        %...
    end

    data = evt.trackM;
    dataname = [evtname, '/trackM'];
    chunksize = size(data);
    chunksize(1) = min(chunksize(1), 250);
    WriteToH5(savename, dataname, data, chunksize);

    this_fn = fieldnames(evt)
    if ~isfield(evt, 'cheat'))
        err = 'No cheat found'
        %...
    end

    % get the name of a pixnoise fieldname for error checks
    a_pixnoise = ''
    for i = 1:length(this_fn)
        if this_fn{i}(1:3) == 'pix'
            a_pixnoise = this_fn{i}
        end
    end
    if a_pixnoise == ''
        err = 'No pixnoise found'
        %...
    end

    if length(evt.cheat) > 1
        err = 'Multiplicity event'
        %...
    end
end

% copied from write_hdf5.m
function WriteToH5(varargin)
    % function WriteToH5(savename, dataname, data, [chunksize])

    savename = varargin{1};
    dataname = varargin{2};
    data = varargin{3};
    compressed = false;
    if nargin==4
        compressed = true;
        chunksize = varargin{4};
    end

    if compressed
        h5create(savename, dataname, size(data), 'ChunkSize', chunksize, 'Deflate', 9);
        h5write(savename, dataname, data);
    else
        h5create(savename, dataname, size(data));
        h5write(savename, dataname, data);
    end
end
