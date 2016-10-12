function DT_to_h5
    % load a DT .mat file
    % save to a DT .h5 file, readable by python
    % Oct. 12, 2016

    loadpath = '/global/home/users/bcplimley/multi_angle/DTbatch01';
    loadpattern = 'MultiAngle_DT_*.mat';

    savepath = '/global/home/users/bcplimley/multi_angle/DTbatch01_h5';

    flist = dir(fullfile(loadpath, loadpattern));

    for i = 1:length(flist)
        % save with same filename as load
        savename = flist(i).name;
        disp(['Starting ', savename, ' at ', datestr(now)])

        fileload = load(fullfile(loadpath,flist(i).name));
        for k = 1:length(fileload.DT)
            % one event
            evtname = ['/', sprintf('%05u', i-1)];

            if isempty(fileload.DT{k})
                err = 'DT cell was empty'
                %...
            end

            data = fileload.DT{k}.trackM;
            dataname = [evtname, '/trackM'];
            chunksize = size(data);
            chunksize(1) = min(chunksize(1), 250);
            WriteToH5(savename, dataname, data, chunksize);

            this_fn = fieldnames(fileload.DT{k})
            if ~isfield(fileload.DT{k}, 'cheat'))
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

            if length(fileload.DT{k}.cheat) > 1
                err = 'Multiplicity event'
                %...
            end


        end
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
