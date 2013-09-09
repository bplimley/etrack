%summarizeDiffusion.m
%
% loads files from Geant4TrackHandling on multiangle simulations, and compiles interesting variables
% (data from /mnt/grimdata5/coffer/eT_MultiAngle...)
% (diffused tracks from /mnt/grimdata5/plimley/multi_angle_2012/...)

% %T400 local
% loadpath = '/mnt/data/Documents/MATLAB/data/Electron Track/geant4/';
% savepath = loadpath;

% %grim
% loadpath = '/mnt/grimdata5/plimley/multi_angle_2012/Batch01/';
% savepath = '/mnt/grimdata5/plimley/multi_angle_2012/summarizeDiffusion/';
% addpath /mnt/grimdata5/plimley/multi_angle_2012/

%lrc
savename = 'summarizeDiffusion5.mat';
loadpath = '/global/scratch/bcplimley/multi_angle/newtracks01out/t5/';
savepath = '/global/scratch/bcplimley/multi_angle/';
addpath /global/scratch/bcplimley/multi_angle/

flist = dir(fullfile(loadpath,'MultiAngle_Diffused_*.mat'));
disp(['Found ',num2str(length(flist)),' files at ',datestr(now)])

%get field names first
load(fullfile(loadpath,flist(1).name),'Event');
% %beware of error on first track
% fieldName = fieldnames(Event{1}.CCD);
% cur = Event{1}.CCD;
% fn2 = fieldnames(Event{2}.CCD);
% if length(fn2) > length(fieldName)
%     fieldName = fn2;
%     cur = Event{2}.CCD;
% end
% clear fn2

%define fieldName by hand to include different fields from different files
fieldName = {...
    'pix2_5noise0',  'pix2_5noise0_003748',  'pix2_5noise0_00937',  'pix2_5noise0_01874',  'pix2_5noise0_03748',  'pix2_5noise0_0937',  'pix2_5noise0_1874',  'pix2_5noise0_3748',  'pix2_5noise0_937', ...
    'pix5noise0',    'pix5noise0_003748',    'pix5noise0_00937',    'pix5noise0_01874',    'pix5noise0_03748',    'pix5noise0_0937',    'pix5noise0_1874',    'pix5noise0_3748',    'pix5noise0_937', ...
    'pix10_5noise0', 'pix10_5noise0_003748', 'pix10_5noise0_00937', 'pix10_5noise0_01874', 'pix10_5noise0_03748', 'pix10_5noise0_0937', 'pix10_5noise0_1874', 'pix10_5noise0_3748', 'pix10_5noise0_937', ...
    'pix20noise0',   'pix20noise0_003748',   'pix20noise0_00937',   'pix20noise0_01874',   'pix20noise0_03748',   'pix20noise0_0937',   'pix20noise0_1874',   'pix20noise0_3748',   'pix20noise0_937', ...
    'pix40noise0',   'pix40noise0_003748',   'pix40noise0_00937',   'pix40noise0_01874',   'pix40noise0_03748',   'pix40noise0_0937',   'pix40noise0_1874',   'pix40noise0_3748',   'pix40noise0_937'};
% NOTE that on line 111, 'pix10_5noise0_01874' is hardcoded because I expect everything to include that...

maxFieldNameLength = 0;
for i=1:length(fieldName)
    if length(fieldName{i}) > maxFieldNameLength
        maxFieldNameLength = length(fieldName{i});
    end
end

%why?
% %get pixsize and noise
% pixStringBase = 'pix';
% noiseStringBase = 'noise';
% fieldPixelSize = nan(1,length(fieldName));
% fieldNoise = nan(1,length(fieldName));
% for i=1:length(fieldName)
%     %pixel size value
%     pixelSizeString = fieldName{i}(length(pixStringBase)+1:strfind(fieldName{i},noiseStringBase)-1);
%     %convert underscores back to decimal point
%     pixelSizeString(strfind(pixelSizeString,'_')) = '.';
%     fieldPixelSize(i) = str2double(pixelSizeString);
%     %noise value
%     noiseString = fieldName{i}(strfind(fieldName{i},noiseStringBase)+length(noiseStringBase):end);
%     %convert underscores back to decimal point
%     noiseString(strfind(noiseString,'_')) = '.';
%     fieldNoise(i) = str2double(noiseString);
% end
% pixelSize = unique(fieldPixelSize);
% noise = unique(fieldNoise);

%initialize
cheat.Etot = nan(1,2*50*length(flist));
cheat.Edep = nan(1,2*50*length(flist));
cheat.alpha = nan(1,2*50*length(flist));
cheat.beta = nan(1,2*50*length(flist));
cheat.Ebrems = nan(1,2*50*length(flist));
cheat.z = nan(1,2*50*length(flist));
cheatIndex = 1; %keep track of where in the variable we are

diffusionIndex = cell(1,length(fieldName));
for k=1:length(fieldName)
    diffusion.(fieldName{k}).Etot = nan(1,50*length(flist));
    diffusion.(fieldName{k}).Edep = nan(1,50*length(flist));
    diffusion.(fieldName{k}).Emeas = nan(1,50*length(flist));
    diffusion.(fieldName{k}).Npix = nan(1,50*length(flist));
    diffusionIndex{k} = 1; %keep track of where in the variable we are
end

disp(['Initialized variables are taking ~',num2str(CheckMemUse(whos)),' bytes'])



%each file
for i=1:length(flist)
    %skip placeholders
    if ~isempty(strfind(flist(i).name,'PH'))
        disp(['Skipping placeholder file ',flist(i).name])
        continue
    end
    %first file already loaded for initialization
    if i>1
        tic;
        load(fullfile(loadpath,flist(i).name),'Event');
        tempTime = toc;
    else
        tempTime = 0;
    end
    disp(['Loaded ',flist(i).name,' at ',datestr(now),' in ',num2str(tempTime),' seconds with ',num2str(CheckMemUse(whos)),' b used'])
    
    %initialize cheat variables - for every single primary electron
    curCheat.Etot = nan(1,length(Event)*2);
    curCheat.Edep = curCheat.Etot;
    curCheat.alpha = curCheat.Etot;
    curCheat.beta = curCheat.Etot;
    curCheat.Ebrems = curCheat.Etot;
    curCheat.z = curCheat.Etot;
    %increment variable for cheat
    n = 1;
    
    %initialize diffusion variables - only for multiplicity = 1 and segments = 1
    for k=1:length(fieldName)
        curDiffusion.(fieldName{k}).Etot = nan(1, length(Event));
        curDiffusion.(fieldName{k}).Edep = curDiffusion.(fieldName{k}).Etot;
        curDiffusion.(fieldName{k}).Emeas = curDiffusion.(fieldName{k}).Etot;
        curDiffusion.(fieldName{k}).Npix = curDiffusion.(fieldName{k}).Etot;
    end
    
    %get this file's fieldnames
    currentFieldNames = fieldnames(Event{1}.CCD);
    if length(fieldnames(Event{2}.CCD)) > length(currentFieldNames)
        currentFieldNames = fieldnames(Event{2}.CCD);
    elseif length(fieldnames(Event{3}.CCD)) > length(currentFieldNames)
        currentFieldNames = fieldnames(Event{3}.CCD);
    end
    
    if isempty(currentFieldNames)
        warning(['Skipping empty Event structure in ',flist(i).name])
    end
    
    for j=1:length(Event)
        if ~isfield(Event{j}.CCD,'pix10_5noise0_01874')
            disp(['Problem on event #',num2str(j),'; field name missing'])
            continue
        end
        %number of primary electrons
        curDataLength = length(Event{j}.CCD.(currentFieldNames{1}).cheat);
            
        %cheat variables (index by n)
        for m=1:length(curDataLength)
            curCheat.Etot(n) = Event{j}.CCD.(currentFieldNames{1}).cheat(m).Etot;
            curCheat.Edep(n) = Event{j}.CCD.(currentFieldNames{1}).cheat(m).Edep;
            curCheat.alpha(n) = Event{j}.CCD.(currentFieldNames{1}).cheat(m).alpha;
            curCheat.beta(n) = Event{j}.CCD.(currentFieldNames{1}).cheat(m).beta;
            curCheat.Ebrems(n) = Event{j}.CCD.(currentFieldNames{1}).cheat(m).Ebrems;
            curCheat.z(n) = Event{j}.CCD.(currentFieldNames{1}).cheat(m).x0(3);
            n=n+1;
        end
        
        for k=1:length(currentFieldNames)   %this file's field names
            if ~any(strncmp(currentFieldNames{k},fieldName,max(length(currentFieldNames{k}),maxFieldNameLength)))
                warning(['Field ',currentFieldNames{k},' not found in list; skipping!'])
                continue
            end
            
            %why do i need these?
%             pixelSizeIndex = find(pixelSize == fieldPixelSize());
%             noiseIndex = find(noise == fieldNoise(i));
            
            if curDataLength==1 && length(Event{j}.CCD.(currentFieldNames{k}).E)==1
                %diffusion variables (index by j)
                curDiffusion.(currentFieldNames{k}).Etot(j) = Event{j}.CCD.(currentFieldNames{k}).cheat.Etot;
                curDiffusion.(currentFieldNames{k}).Edep(j) = Event{j}.CCD.(currentFieldNames{k}).cheat.Edep;
                curDiffusion.(currentFieldNames{k}).Emeas(j) = Event{j}.CCD.(currentFieldNames{k}).E;
                curDiffusion.(currentFieldNames{k}).Npix(j) = sum(Event{j}.CCD.(currentFieldNames{k}).T{1}.img(:) ~=0);
            end
            
            
        end
    end
    
    %clean cheat variables
    if n <= length(Event)*2
        curCheat.Etot(n:end) = [];
        curCheat.Edep(n:end) = [];
        curCheat.alpha(n:end) = [];
        curCheat.beta(n:end) = [];
        curCheat.Ebrems(n:end) = [];
        curCheat.z(n:end) = [];
    end
    
    
    for k=1:length(fieldName)   %all field names
        %clean diffusion variables
        lg = ~isnan(curDiffusion.(fieldName{k}).Etot);
        curDiffusion.(fieldName{k}).Etot = curDiffusion.(fieldName{k}).Etot(lg);
        curDiffusion.(fieldName{k}).Edep = curDiffusion.(fieldName{k}).Edep(lg);
        curDiffusion.(fieldName{k}).Emeas = curDiffusion.(fieldName{k}).Emeas(lg);
        curDiffusion.(fieldName{k}).Npix = curDiffusion.(fieldName{k}).Npix(lg);
        
        %add diffusion to full diffusion variables
        curIndices = diffusionIndex{k} : diffusionIndex{k} - 1 + length(curDiffusion.(fieldName{k}).Etot);
        
        diffusion.(fieldName{k}).Etot(curIndices) = curDiffusion.(fieldName{k}).Etot;
        diffusion.(fieldName{k}).Edep(curIndices) = curDiffusion.(fieldName{k}).Edep;
        diffusion.(fieldName{k}).Emeas(curIndices) = curDiffusion.(fieldName{k}).Emeas;
        diffusion.(fieldName{k}).Npix(curIndices) = curDiffusion.(fieldName{k}).Npix;
        
        %increment
        diffusionIndex{k} = diffusionIndex{k} + length(curDiffusion.(fieldName{k}).Etot);
    end
    
    %add cheat to full cheat variables
    curIndices = cheatIndex : cheatIndex - 1 + length(curCheat.Etot);
    
    cheat.Etot(curIndices) = curCheat.Etot;
    cheat.Edep(curIndices) = curCheat.Edep;
    cheat.alpha(curIndices) = curCheat.alpha;
    cheat.beta(curIndices) = curCheat.beta;
    cheat.Ebrems(curIndices) = curCheat.Ebrems;
    cheat.z(curIndices) = curCheat.z;
    
    cheatIndex = cheatIndex + length(curCheat.Etot);
    
    disp(['Finished ',flist(i).name,' at ',datestr(now)])
end

%clean final cheat
cheat.Etot(cheatIndex:end) = [];
cheat.Edep(cheatIndex:end) = [];
cheat.alpha(cheatIndex:end) = [];
cheat.beta(cheatIndex:end) = [];
cheat.Ebrems(cheatIndex:end) = [];
cheat.z(cheatIndex:end) = [];

%clean final diffusion
for k=1:length(fieldName)
    diffusion.(fieldName{k}).Etot(diffusionIndex{k}:end) = [];
    diffusion.(fieldName{k}).Edep(diffusionIndex{k}:end) = [];
    diffusion.(fieldName{k}).Emeas(diffusionIndex{k}:end) = [];
    diffusion.(fieldName{k}).Npix(diffusionIndex{k}:end) = [];
end
save(fullfile(savepath,savename),'cheat','diffusion','flist')
