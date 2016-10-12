function write_DT_hdf5(loadfile, savename, donefilename)
    % function write_DT_hdf5(loadfile,savename)
    %
    % Load from MAT, write to HDF5.
    %
    % copied from write_hdf5.m.
    %
    % for DT mat-files instead of HT mat-files.
    % the file structures are very similar, but DT has a T{} cell array, while
    %   HT unpacks that because there can only be one track image to
    %   run the algorithm on.

    progressflag = false;
    donefileflag = true;
    if ischar(loadfile)
        filedata = load(loadfile);
    else
        filedata = loadfile;
    end
    fn = fieldnames(filedata);
    if any(strcmp(fn,'HT'))
        tracks = filedata.HT;
    elseif any(strcmp(fn,'Event'))
        tracks = filedata.Event;
    end

    if progressflag
        progressbar(0);
    end
    % assume tracks is a cell array
    for i=1:length(tracks)
        dataname = ['/', sprintf('%05u', i-1)];

        writeHT(tracks{i}, dataname, savename);
        progressbar(i/length(tracks));
    end

    if donefileflag
        save(donefilename, 'donefileflag')
    end
end

% ############################################################################

function writeHT(track, parentname, savename)
    % function writeHT(track, dataname, savename)
    if isempty(track) || ~isstruct(track)
        return
    end

    if ~isfield(track, 'pix10_5noise0') || ...
            ~isfield(track, 'Etot') || ...
            ~isfield(track, 'multiplicity') || ...
            ~isfield(track, 'trackM')
        return
    end

    % First, write the track matrix which should always be there, in order to
    %   get the file and group to exist.
    data = track.trackM;
    dataname = [parentname, '/trackM'];
    chunksize = size(data);
    chunksize(1) = min(chunksize(1), 250);
    WriteToH5(savename, dataname, data, chunksize);

    % other top-level attributes for any multiplicity
    attname = 'multiplicity';
    attvalue = track.multiplicity;
    CheckScalar(attvalue);
    h5writeatt(savename, parentname, attname, attvalue);

    errorcode = 0;
    if isfield(track, 'out') && isfield(track.out, 'err') && ...
            ~isempty(track.out.err)
        switch track.out.err
            case ''
                errorcode = 1;
            otherwise
                error(['new error in writeHT: ', track.out.err])
        end
        attname = 'out_errcode';
        attvalue = errorcode;
        h5writeatt(savename, parentname, attname, attvalue);
    end

    attname = 'Etot';
    attvalue = track.Etot;
    if length(attvalue) ~= track.multiplicity
        error('Etot wrong length')
    end
    h5writeatt(savename, parentname, attname, attvalue);

    attname = 'Edep';
    attvalue = track.Edep;
    if length(attvalue) ~= track.multiplicity
        error('Edep wrong length')
    end
    h5writeatt(savename, parentname, attname, attvalue);

    attname = 'Eesc';
    attvalue = track.Eesc;
    if length(attvalue) ~= track.multiplicity
        error('Eesc wrong length')
    end
    h5writeatt(savename, parentname, attname, attvalue);

    fn = fieldnames(track);

    if track.multiplicity == 1
        dataname = [parentname, '/cheat/0'];
        writeCheat(track.cheat, dataname, savename)
    elseif track.multiplicity > 1
        for i = 1:track.multiplicity
            dataname = [parentname, '/cheat/', num2str(i-1)];
            writeCheat(track.cheat, dataname, savename)
        end
    end

    for i = 1:length(fn)
        if strcmp(fn{i}(1:3), 'pix')
            % this is a pix*noise* field
            % TODO: check for bad state of this pixnoise...

            dataname = [parentname, '/', fn{i}];
            pixnoise = track.(fn{i});
            writePixNoise(pixnoise, fn{i}, dataname, savename, ...
                track.multiplicity)
        end
    end
end

% ############################################################################

function writeCheat(cheat, parentname, savename)
    % function writeCheat(cheat, parentname, savename)

    dataname = [parentname,'/dE'];
    data = cheat.dE;
    WriteToH5(savename, dataname, data);    % dataset

    dataname = [parentname,'/x'];
    data = cheat.x;
    WriteToH5(savename, dataname, data);    % dataset

    dataname = parentname;

    attname = 'Etot';
    attvalue = cheat.Etot;
    CheckScalar(attvalue);
    h5writeatt(savename, parentname, attname, attvalue);

    attname = 'Edep';
    attvalue = cheat.Edep;
    CheckScalar(attvalue);
    h5writeatt(savename, parentname, attname, attvalue);

    attname = 'Exray';
    attvalue = cheat.Exray;
    CheckScalar(attvalue);
    h5writeatt(savename, parentname, attname, attvalue);

    attname = 'Ebrems';
    attvalue = cheat.Ebrems;
    CheckScalar(attvalue);
    h5writeatt(savename, parentname, attname, attvalue);

    attname = 'E0';
    attvalue = cheat.E0;
    CheckScalar(attvalue);
    h5writeatt(savename, parentname, attname, attvalue);

    attname = 'particleID';
    attvalue = cheat.particleID;
    CheckScalar(attvalue);
    h5writeatt(savename, parentname, attname, attvalue);

    attname = 'x0';
    attvalue = cheat.x0;
    if length(attvalue) ~= 3
        error('Bad x0')
    end
    h5writeatt(savename, parentname, attname, attvalue);

    attname = 'firstStepVector';
    attvalue = cheat.firstStepVector;
    if length(attvalue) ~= 3
        error('Bad firstStepVector')
    end
    h5writeatt(savename, parentname, attname, attvalue);

    attname = 'longStepLength';
    attvalue = cheat.longStepLength;
    CheckScalar(attvalue);
    h5writeatt(savename, parentname, attname, attvalue);

    attname = 'alpha';
    attvalue = cheat.alpha;
    CheckScalar(attvalue);
    h5writeatt(savename, parentname, attname, attvalue);

    attname = 'alphaLong';
    attvalue = cheat.alphaLong;
    CheckScalar(attvalue);
    h5writeatt(savename, parentname, attname, attvalue);

    attname = 'beta';
    attvalue = cheat.beta;
    CheckScalar(attvalue);
    h5writeatt(savename, parentname, attname, attvalue);

    attname = 'betaLong';
    attvalue = cheat.betaLong;
    CheckScalar(attvalue);
    h5writeatt(savename, parentname, attname, attvalue);

    attname = 'XrayDistance';
    attvalue = cheat.XrayDistance;
    CheckScalar(attvalue);
    h5writeatt(savename, parentname, attname, attvalue);

    attname = 'sourcePhotonE1';
    attvalue = cheat.sourcePhotonE1;
    CheckScalar(attvalue);
    h5writeatt(savename, parentname, attname, attvalue);

    attname = 'sourcePhotonE2';
    attvalue = cheat.sourcePhotonE2;
    CheckScalar(attvalue);
    h5writeatt(savename, parentname, attname, attvalue);

    attname = 'sourcePhotonDirection1';
    attvalue = cheat.sourcePhotonDirection1;
    if length(attvalue) ~= 3
        error('Bad sourcePhotonDirection1')
    end
    h5writeatt(savename, parentname, attname, attvalue);

    attname = 'sourcePhotonDirection2';
    attvalue = cheat.sourcePhotonDirection2;
    if length(attvalue) ~= 3
        error('Bad sourcePhotonDirection2')
    end
    h5writeatt(savename, parentname, attname, attvalue);
end

% ############################################################################

function writeT(track, parentname, savename)
    % function writeT(track, dataname, savename)

    dataname = [parentname, '/img'];

    data = track.img';  % MATLAB transposes somehow
    WriteToH5(savename, dataname, data, size(data));  % compressed

    dataname = parentname;

    attname = 'E';
    attvalue = track.E;
    CheckScalar(attvalue);
    h5writeatt(savename, dataname, attname, attvalue);

    attname = 'x';
    attvalue = track.x;
    CheckScalar(attvalue);
    h5writeatt(savename, dataname, attname, attvalue);

    attname = 'y';
    attvalue = track.y;
    CheckScalar(attvalue);
    h5writeatt(savename, dataname, attname, attvalue);

    attname = 'edgeflag';
    attvalue = +track.edgeflag;
    CheckScalar(attvalue);
    h5writeatt(savename, dataname, attname, attvalue);
end

% ############################################################################

function writePixNoise(pixnoise, fieldname, parentname, savename, multiplicity)
    % function writePixNoise(pixnoise, fieldname, parentname, savename,
    %   multiplicity)

    % first, assign error code so we know what to do later
    if ~isfield(pixnoise, 'err') && multiplicity == 1
        % normal
        errorcode = 0;
    elseif ~isfield(pixnoise, 'err') && multiplicity > 1
        % multiplicity event. skipped in previous algorithm batch.
        errorcode = 6;
    elseif multiplicity == 0
        error('multiplicity of 0, does this ever happen?')
    elseif strcmpi(pixnoise.err, 'DT pixnoise did not exist')
        errorcode = 1;
    elseif strcmpi(pixnoise.err, 'Segmentation divided the track')
        errorcode = 2;
    elseif strcmpi(pixnoise.err, 'No segmented image to use')
        errorcode = 3;
    elseif strcmpi(pixnoise.err, 'No ends found')
        errorcode = 4;
    elseif strcmpi(pixnoise.err, 'Infinite loop')
        errorcode = 5;
    elseif strcmpi(class(pixnoise.err), 'MException') && ...
            strcmpi(pixnoise.err.identifier, 'MATLAB:nonLogicalConditional')
        errorcode = 7;
    else
        error('newerror', ['new error: ', pixnoise.err])
    end

    algorithm_error = (errorcode == 4 || errorcode == 5);
    good_algorithm = (errorcode == 0);
    has_algorithm = good_algorithm || algorithm_error;
    has_ridge = good_algorithm;
    has_measurement = good_algorithm;
    has_multiple_tracks = (errorcode == 2 || errorcode == 6);

    do_pixnoise = true;
    check_pixsize = good_algorithm || has_multiple_tracks;
    check_noise = has_multiple_tracks;
    do_img = has_algorithm;

    do_EtotTind = has_algorithm;
    do_nends = good_algorithm || errorcode == 4;

    do_T = has_multiple_tracks;
    do_edgesegments = good_algorithm;
    do_ridge = has_ridge;
    do_measurement = has_measurement;


    [pixsize, noise] = GetPixNoise(fieldname);

    dataname = parentname;
    % force h5group creation
    WriteToH5(savename, [dataname, '/asdf'], 42);

    attname = 'errorcode';
    attvalue = errorcode;
    h5writeatt(savename, dataname, attname, attvalue);

    if do_pixnoise
        attname = 'pixel_size_um';
        attvalue = pixsize;
        if check_pixsize && attvalue ~= pixnoise.pixsize
            error('Pixel size mismatch')
        end
        CheckScalar(attvalue);
        h5writeatt(savename, dataname, attname, attvalue);

        attname = 'noise_ev';
        attvalue = noise;
        if check_noise && attvalue ~= pixnoise.noise * 1000
            error('noise labeling mismatch')
        end
        CheckScalar(attvalue);
        h5writeatt(savename, dataname, attname, attvalue);
    end

    if do_img
        dataname = [parentname, '/img'];
        data = pixnoise.img';  % MATLAB transposes somehow
        WriteToH5(savename, dataname, data, size(data));  % compressed
    end

    if do_T
        dataname = parentname;

        for i = 1:length(pixnoise.T)
            writeT(pixnoise.T{i}, [dataname, '/T', num2str(i-1)], savename)
        end

        attname = 'pixel_size_um';
        attvalue = pixsize;
        CheckScalar(attvalue);
        h5writeatt(savename, dataname, attname, attvalue);

        attname = 'noise_ev';
        attvalue = noise;
        CheckScalar(attvalue);
        h5writeatt(savename, dataname, attname, attvalue);

        attname = 'segment_threshold';
        attvalue = pixnoise.segmentThreshold;
        CheckScalar(attvalue);
        h5writeatt(savename, dataname, attname, attvalue);

        attname = 'pixel_threshold';
        attvalue = pixnoise.pixelThreshold;
        CheckScalar(attvalue);
        h5writeatt(savename, dataname, attname, attvalue);

        attname = 'E';
        attvalue = pixnoise.E;
        h5writeatt(savename, dataname, attname, attvalue);

        if length(pixnoise.E) ~= length(pixnoise.T)
            error('Energy and track list length mismatch')
        end
    end

    if do_EtotTind
        dataname = parentname;

        attname = 'Etot';
        attvalue = pixnoise.Etot;
        CheckScalar(attvalue);
        h5writeatt(savename, dataname, attname, attvalue);

        attname = 'Tind';
        attvalue = pixnoise.Tind;
        CheckScalar(attvalue);
        h5writeatt(savename, dataname, attname, attvalue);
    end

    if do_nends
        dataname = parentname;

        attname = 'n_ends';
        attvalue = pixnoise.ends;
        CheckScalar(attvalue);
        h5writeatt(savename, dataname, attname, attvalue);
    end

    if do_edgesegments
        dataname = parentname;

        attname = 'lt';
        attvalue = pixnoise.lt;
        CheckScalar(attvalue);
        h5writeatt(savename, dataname, attname, attvalue);

        attname = 'Eend';
        attvalue = pixnoise.Eend;
        CheckScalar(attvalue);
        h5writeatt(savename, dataname, attname, attvalue);

        attname = 'edgesegments_energies_kev';
        attvalue = pixnoise.EdgeSegments.energiesKev;
        L = length(attvalue);
        h5writeatt(savename, dataname, attname, attvalue);

        attname = 'edgesegments_coordinates_pix';
        attvalue = pixnoise.EdgeSegments.coordinatesPix;
        if size(attvalue, 1) ~= L || size(attvalue, 2) ~= 2
            error('bad size of coordinates_pix')
        end
        h5writeatt(savename, dataname, attname, attvalue);

        attname = 'edgesegments_chosen_index';
        attvalue = pixnoise.EdgeSegments.chosenIndex;
        CheckScalar(attvalue);
        h5writeatt(savename, dataname, attname, attvalue);

        attname = 'edgesegments_start_coordinates_pix';
        attvalue = pixnoise.EdgeSegments.startCoordinatesPix;
        if length(attvalue) ~= 2
            error('bad size of start_coordinates_pix')
        end
        h5writeatt(savename, dataname, attname, attvalue);

        attname = 'edgesegments_start_direction_indices';
        attvalue = pixnoise.EdgeSegments.startDirectionIndices;
        CheckScalar(attvalue);
        h5writeatt(savename, dataname, attname, attvalue);

        attname = 'edgesegments_low_threshold_used';
        attvalue = pixnoise.EdgeSegments.lowThresholdUsed;
        CheckScalar(attvalue);
        h5writeatt(savename, dataname, attname, attvalue);
    end

    if do_ridge
        dataname = [parentname, '/thinned_img'];
        data = pixnoise.thin';  % MATLAB transposes somehow
        WriteToH5(savename, dataname, data, size(data));  % compressed

        dataname = [parentname, '/x'];
        data = pixnoise.x;
        WriteToH5(savename, dataname, data, size(data));  % compressed

        dataname = [parentname, '/y'];
        data = pixnoise.y;
        WriteToH5(savename, dataname, data, size(data));  % compressed

        dataname = [parentname, '/w'];
        data = pixnoise.w;
        WriteToH5(savename, dataname, data, size(data));  % compressed

        dataname = [parentname, '/a0'];
        data = pixnoise.a0;
        WriteToH5(savename, dataname, data, size(data));  % compressed

        dataname = [parentname, '/dE'];
        data = pixnoise.dE;
        WriteToH5(savename, dataname, data, size(data));  % compressed
    end

    if do_measurement
        dataname = parentname;

        attname = 'alpha';
        attvalue = pixnoise.alpha;
        CheckScalar(attvalue);
        h5writeatt(savename, dataname, attname, attvalue);

        attname = 'beta';
        attvalue = pixnoise.beta;
        CheckScalar(attvalue);
        h5writeatt(savename, dataname, attname, attvalue);

        attname = 'dalpha';
        attvalue = pixnoise.dalpha;
        CheckScalar(attvalue);
        h5writeatt(savename, dataname, attname, attvalue);

        attname = 'dbeta';
        attvalue = pixnoise.dbeta;
        CheckScalar(attvalue);
        h5writeatt(savename, dataname, attname, attvalue);

        attname = 'dedx_ref';
        attvalue = pixnoise.Measurement.dedxReference;
        CheckScalar(attvalue);
        h5writeatt(savename, dataname, attname, attvalue);

        attname = 'dedx_meas';
        attvalue = pixnoise.Measurement.dedxMeasured;
        CheckScalar(attvalue);
        h5writeatt(savename, dataname, attname, attvalue);

        attname = 'measurement_start_ind';
        attvalue = pixnoise.Measurement.indices(1);
        CheckScalar(attvalue);
        h5writeatt(savename, dataname, attname, attvalue);

        attname = 'measurement_end_ind';
        attvalue = pixnoise.Measurement.indices(end);
        CheckScalar(attvalue);
        h5writeatt(savename, dataname, attname, attvalue);
    end

end

function [pixsize, noise] = GetPixNoise(fname)
    % function [pixsize, noise] = GetPixNoise(fname)

    noiselocation = strfind(fname,'noise');
    pixstring = fname(4:noiselocation-1);
    noisestring = fname(noiselocation+5:end);

    pixstring = strrep(pixstring,'_','.');
    noisestring = strrep(noisestring,'_','.');

    pixsize = str2double(pixstring);
    noise = str2double(noisestring);
end

function CheckScalar(data)
    % function CheckScalar(data)

    if length(data) == 0
        error('Empty data')
    elseif length(data) > 1
        error('Too many data values')
    else
        return
    end
end

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
