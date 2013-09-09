function MultiAngle_worker(loadpath,flist,savepath,savepre,loadpre,loadsuf,savesuf,noise,pixelSize,pixelNoiseMatrix,i,placeholdersuf,psft)
fileload = load(fullfile(loadpath,flist(i).name));

addpath /global/home/users/bcplimley/improc/
addpath /global/home/users/bcplimley/cfit/

%batches of 1000, then save off
batchsize = 50;
    for j=1:ceil(length(fileload.Event)/batchsize)
        numstring = [flist(i).name(length(loadpre)+1:strfind(flist(i).name,loadsuf)-1),'_',num2str(j)];  %includes flist number as well as part number
        savename = [savepre,numstring,savesuf];
        placeholdername = [savepre,numstring,placeholdersuf];
        if ~isempty(dir(fullfile(savepath,savename))) || ~isempty(dir(fullfile(savepath,placeholdername)))
            disp(['Skipping ',savename,' at ',datestr(now)])
            continue
        end
        
        %write placeholder file
        save(fullfile(savepath,placeholdername),'numstring');
        
        disp(['Starting ',savename,' at ',datestr(now),' with ',num2str(CheckMemUse(whos)),' bytes in memory'])
        
        %each event
        for k = (j-1)*batchsize+1:min(j*batchsize,length(fileload.Event))
            filesave.Event{k-(j-1)*batchsize} = fileload.Event{k};
            try
                filesave.Event{k-(j-1)*batchsize}.CCD = Geant4TrackHandling7(fileload.Event{k}.trackM,'coinc','driftdim','z','depthcoordinates',[-0.65,0],'noise',noise,'pixelsize',pixelSize,'pixelnoisematrix',pixelNoiseMatrix,'psft',psft);
            catch err
                filesave.Event{k-(j-1)*batchsize}.CCD.err = err;
                disp(['Warning: encountered error "',err.message,'" in ',flist(i).name,' event # ',num2str(k)])
            end
        end

        %save
%        disp(['Saving ',savename,' at ',datestr(now)])
        savefile(fullfile(savepath,savename),filesave.Event,flist);
        disp(['Finished ',savename,' at ',datestr(now),' with ',num2str(CheckMemUse(whos)),' bytes in memory'])
        %remove placeholder
        delete(fullfile(savepath,placeholdername));
    end

