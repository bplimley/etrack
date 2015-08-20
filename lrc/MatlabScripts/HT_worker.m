function HT_worker(loadpath,flist,savepath,savepre,loadpre,loadsuf,savesuf,noise,pixelSize,fieldNames,i,placeholdersuf,dEdx_ref)

fileload = load(fullfile(loadpath,flist(i).name));

% %batches of 1000, then save off
% batchsize = 50;
%    for j=1:length(fileload.Event)
        
        %numstring = [flist(i).name(length(loadpre)+1:strfind(flist(i).name,loadsuf)-1),'_',num2str(j)];  
        numstring = [flist(i).name(length(loadpre)+1:strfind(flist(i).name,loadsuf)-1)];
        %   includes flist number as well as part number
        savename = [savepre,numstring,savesuf];
        placeholdername = [savepre,numstring,placeholdersuf];
        if ~isempty(dir(fullfile(savepath,savename)))
            disp(['Skipping ',savename,' at ',datestr(now)])
            return
        elseif ~isempty(dir(fullfile(savepath,placeholdername)))
            disp(['Skipping ',placeholdername,' at ',datestr(now)])
            return
        end
        
        %write placeholder file
        save(fullfile(savepath,placeholdername),'numstring');
        
%        disp(['Starting ',savename,' at ',datestr(now),' with ',num2str(CheckMemUse(whos)),' bytes in memory'])
        disp(['Starting ',savename,' at ',datestr(now)])
        %each event
        for k = 1:length(fileload.Event)
            filesave.Event{k} = fileload.Event{k};
            %skip multiplicity events for simplicity
            if length(fileload.Event{k}.CCD.(fieldNames{1}).cheat) > 1
                continue
            end
            for m=1:length(fieldNames)
                %need to match cheat events to diffused images
                %   use lateral position...
                
                %do this for reals later. I am going to save time by taking only simple events.
                if length(fileload.Event{k}.CCD.(fieldNames{m}).T) > 1
                    %note this as an efficiency loss!
                    filesave.Event{k}.HT.err = 'Segmentation divided the track';
                    continue
                end
                
                %from Geant4TrackHandling
                segmentThresholdBase = 0.06;    %not an argument yet..
                segmentThresholdSlope = (0.5 - segmentThresholdBase) / 10.5^2;   % keV/um^2 of pixelSize
                segmentThresholdCurrentBase = segmentThresholdBase / 0.01874 * noise(m);
                segmentThresholdCurrent = segmentThresholdCurrentBase + ...
                    segmentThresholdSlope * pixelSize(m)^2;
                %note from TrackHandling for CCDsegment3:   set so that 10.5 um pixels give a standard 0.55 keV threshold
                % HybridTrack_1b says I used 0.5 keV threshold since 2009, so I adjusted Slope definition accordingly.
                
                try
                    if isempty(fileload.Event{k}.CCD.(fieldNames{m}).T)
                        %no segmented image: e.g. low Edep, back side, small pixels
                        filesave.Event{k}.HT.err = 'No segmented image to use';
                        continue
                    end
                    tic;
                    filesave.Event{k}.HT.(fieldNames{m}) = HybridTrack_1b(...
                        fileload.Event{k}.CCD.(fieldNames{m}).T{1}.img, ...  %img
                        segmentThresholdCurrent, ...%lowThresh
                        pixelSize(m), ... %pixelSize
                        dEdx_ref, ...    %dEref
                        fileload.Event{k}.CCD.(fieldNames{m}).cheat, ...    %cheat    
                        false);    %plotflag
                    %disp([fieldNames{m},' took ',num2str(toc),' seconds'])
                    %calculate dalpha, dbeta
                    if isfield(filesave.Event{k}.HT.(fieldNames{m}),'alpha')
                        filesave.Event{k}.HT.(fieldNames{m}).dalpha = ...
                            filesave.Event{k}.HT.(fieldNames{m}).alpha - ...
                            fileload.Event{k}.CCD.(fieldNames{m}).cheat.alpha;
                        filesave.Event{k}.HT.(fieldNames{m}).dbeta = ...
                            filesave.Event{k}.HT.(fieldNames{m}).beta - ...
                            fileload.Event{k}.CCD.(fieldNames{m}).cheat.beta;
                    else
                        filesave.Event{k}.HT.(fieldNames{m}).dalpha = nan;
                        filesave.Event{k}.HT.(fieldNames{m}).dbeta = nan;
                    end
                catch err
                    filesave.Event{k}.HT.err = err;
                    disp(['Warning: encountered error "',err.message,'" in ',flist(i).name,' event # ',num2str(k)])
                end
            end
        end

        %save
%        disp(['Saving ',savename,' at ',datestr(now)])
        savefile(fullfile(savepath,savename),filesave.Event,flist);
        disp(['Finished ',savename,' at ',datestr(now),' with ',num2str(CheckMemUse(whos)),' bytes in memory'])
        clear('filesave','fileload')
        %remove placeholder
        delete(fullfile(savepath,placeholdername));
 %   end

