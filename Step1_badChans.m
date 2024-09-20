%% ======================================= getChannelVars ==========================================
close all; clear all; clc

whichGroup = 'ADHD';%'Control';'ADHD'

% Computes and saves the channel variances for each block, so these
% can be plotted to look for bad channels using check4badchans.m.
% Variance calc from FFT spectrum so can avoid certain frequencies.
% list ALL data files in a study in a cell array of cell arrays such that
% files{s}{n} is the name of the nth file of session (subject) s.
% also list session IDs (subject initials or whatever) in cell array sessionID.
% Note: Some participants are not part of this project and will be removed in the next steps

cd('/Users/manab/GoogleDrive_Gmail/DecisionMaking/')
addpath(genpath('/Users/manab/Desktop/Functions/eeglab2021.0/'));
eeglab;

if strcmpi(whichGroup, 'ADHD')
    %% ---------------------------------------------------------------------------------------------
    % ADHD
    % ----------------------------------------------------------------------------------------------

    datafolder = '/Volumes/shared/R-MNHS-SPP/Bellgrove-data/Mana/data/Raw/ADHD/';
    matfolder =  '/Volumes/shared/R-MNHS-SPP/Bellgrove-data/Mana/data/Raw/ADHD/';

    subject_folder = {'AD88C' 'AD91C' 'AD92C' 'AD93C' 'AD94C' 'AD96C' 'AD33C' 'AD34C' 'AD35C' 'AD36C' 'AD37C' 'AD81C' 'AD82C' 'AD83C' 'AD84C' 'AD85C'  'AD86C' ...
        'AD40C' 'AD11C' 'AD43C' 'AD18C' 'AD24C' 'AD52C' 'AD5C'...
        'AD54C' 'AD75C' 'AD15C' 'AD49C' 'AD69C' 'AD32C' 'AD16C' 'AD72C'...
        'AD59C' 'AD27C' 'AD46C' 'AD22C' 'AD26C' 'AD99C' 'AD48C' 'AD57C'...
        'AD51C' 'AD19C' 'AD98C' 'AD25C' 'AD23C' };

    sessionID = {'1_88' '1_91' '1_92' '1_93' '1_94' '1_96' '1_33' '1_34' '1_35' '1_36' '1_37' '1_81' '1_82' '1_83' '1_84' '1_85' '1_86'...
        '1_40' '1_11' '1_43' '1_18' '1_24' '1_52' '1_5'...
        '1_54' '1_75' '1_15' '1_49' '1_69' '1_32' '1_16'...
        '1_72' '1_59' '1_27' '1_46' '1_22' '1_26' '1_99' '1_48' '1_57'...
        '1_51' '1_19' '1_98' '1_25' '1_23'};

    monashSessionID = {'1_88' '1_91' '1_92' '1_93' '1_94' '1_96' '1_33' '1_34' '1_35' '1_36' '1_37' '1_81' '1_82' '1_83' '1_84' '1_85' '1_86'};

    allBlocks = {[1:10] [1:10] [1:10] [1:10] [1:10] [1:5 7:10] [1:10] [1:10] [1:10] [1:10] [1:10] [1:6] [1:10] [1:5] [1:10] [1:10] [1:10]...
        [1:10] [1:10] [1:7] [1:10] [1:10] [1:10] [1:10] [1:10] [1:10] ...
        [1:10] [1:10] [1:10] [1:11] [1:10] [1:10] [1:10] [1:10] [1:10] ...
        [1:10] [1:10] [1:9] [1:10] [1:10] [1:10] [1:11] [1:10] [1:10] [1:10]};
elseif strcmpi(whichGroup, 'ADHD')
    %% ---------------------------------------------------------------------------------------------
    % Control
    % ----------------------------------------------------------------------------------------------
    datafolder = '/Volumes/shared/R-MNHS-SPP/Bellgrove-data/Mana/data/Raw/Control/';
    matfolder =  '/Volumes/shared/R-MNHS-SPP/Bellgrove-data/Mana/data/Raw/Control/';

    subject_folder = {'C133' 'C134' 'C135' 'C136' 'C137' 'C138' 'C139' 'C141' 'C500' 'C501' 'C502' ...
        'C503' 'C504' 'C505' 'C506' 'C507' 'C508' 'C509' 'C510' 'C511' 'C512' '001_KK' 'C12' 'C21' ...
        '002_LK' 'C50' 'C131' 'C20' 'C132' 'C213' 'C204' 'C171' 'C13' 'C119' 'C121' 'C194' 'C168' ...
        'C14' 'C49' 'C400' 'C100' 'C117' 'C115' 'C110' 'C183' 'C144' 'C252' 'C140' 'C259' 'C143' ...
        'C89' 'C62' 'C88' 'C84' 'C87'};

    sessionID = {'133' '134' '135' '136' '137' '138' '139' '141' '500' '501' '502' ...
        '503' '504' '505' '506' '507' '508' '509' '510' '511' '512' '001' '12' '21' ...
        '002' '50' '131' '20' '132' '213' '204' '171' '13' '119' '121' '194' '168' ...
        '14' '49' '400' '100' '117' '115' '110' '183' '144' '252' '140' '259' '143' ...
        '89' '62' '88' '84' '87'};

    monashSessionID = {'133' '134' '135' '136' '137' '138' '139' '141' '500' '501' ...
        '502' '503' '504' '505' '506' '507' '508' '509' '510' '511' '512'};


    allBlocks = {[1:10] [1:10] [1:10] [1:10] [1:10] [1:5] [1:10] [1:10] [1:5] [1:5] [1:10] ...
        [1:10] [1:10] [1:10] [1:10] [1:10] [1:5] [1:10] [1:5] [1:10] [1:10] [1:11] [1:10] ...
        [1:10] [1:10] [1:10] [1:10] [1:10] [1:10] [1:10] [1:10] [1:10] [1:10] [1:10] [1:10] [1:10] ...
        [1:10] [1:10] [1:10] [1:10] [1:4 6:11] [1:10] [1:10] [1:10] [1:10] [1:10] [1:10] [1:10] [1:10] ...
        [1:10] [1:10] [1:10] [1:10] [1:10] [1:10]};
end

    exp_conditions = {'C'}; c = 1;
    file_start = 1;

    clear files matfile
    for s = file_start:length(subject_folder)
        f = 0;
        for b =  allBlocks{s}
            f=f+1;
            if ismember(sessionID{s},monashSessionID)
                files{s}{f} = [datafolder subject_folder{s} '/' sessionID{s} '_' exp_conditions{c} num2str(b) '.vhdr'];
            else
                files{s}{f} = [datafolder subject_folder{s} '/' sessionID{s} '_' exp_conditions{c} num2str(b) '.bdf'];
            end
            matfile{s} = [matfolder subject_folder{s} '/' sessionID{s} '_' exp_conditions{c} '_chanvars.mat'];
        end
    end


    % How much of the spectrum to use
    speclims = [4 48];  % Limits in Hz

    minBreakDur_s = 25;  % number of seconds defining a "break" between allBlocks within a data file.
    minBlockDur_s = 60;  % a block must be at least 30s, otherwise the pause btw triggers is something else.

%% -------------------------------------------------------------------------------------------------
% Get Channels Var
% --------------------------------------------------------------------------------------------------

    h = waitbar(0,'Please wait...');
    steps = length(sessionID);
    step = file_start-1;
    for s = file_start%:length(subject_folder) %change file_start to next one when error
        if ismember(sessionID{s},monashSessionID)
            chanlocs = load('ADHD_chanlocs.mat');
            chanlocs = chanlocs.chanlocs;
        else
            chanlocs = readlocs('cap64.loc');
        end

        disp(s)
        tic

        step = step+1;
        numb = 0; clear files1 blockrange;

        for f =1:length(files{s})
            if ismember(sessionID{s},monashSessionID)
                [ filepath , name , ext ] = fileparts(files{s}{f});
                EEG = pop_loadbv([filepath,'/'], [name , ext]);
                clear  filepath  name   ext
            else
                EEG = pop_biosig(files{s}{f}); % read in just one channel to get triggers
            end


            if EEG.srate>500
                EEG = pop_resample(EEG, 500);
            end
            % Fish out the event triggers and times
            clear trigs stimes
            for i = 1:length(EEG.event)
                if ~isa(EEG.event(i).type,'double') && ~startsWith(EEG.event(i).type,'S')
                    EEG.event(i).type = [];
                    EEG.event(i).type = 0;
                elseif  ~isa(EEG.event(i).type,'double') && startsWith(EEG.event(i).type,'S')
                    EEG.event(i).type = regexp(EEG.event(i).type,'\d*','Match');
                    EEG.event(i).type = str2num(EEG.event(i).type{1});
                end
                trigs(i) = EEG.event(i).type;
                stimes(i) = EEG.event(i).latency;

            end

            % %         figure
            % %         plot(trigs)
            %
            minBreakDur = minBreakDur_s*EEG.srate;  % >2s between triggers => inter-block break
            brk = find(diff(stimes)>minBreakDur);
            brkInd = find([EEG.event(brk).type]) ~=0; % Only look at task triggers


            if ~isempty(brk) && ~isempty(brkInd)
                breaktimes_samp = (stimes(brk(brkInd))+stimes(brk(brkInd)+1))/2;    % break times in sample points
            else
                breaktimes_samp = [];
            end


            breaktimes_s = [0 breaktimes_samp size(EEG.data,2)]./EEG.srate;    % break times in sec
            breaktimes_s(find(diff(breaktimes_s)<minBlockDur_s)) = [];  % get rid of breaks that separate segments of data that are too short to be a block
            for b=1:length(breaktimes_s)-1
                numb=numb+1;
                files1{numb} = files{s}{f};
                blockrange{numb} = breaktimes_s(b:b+1);
            end
        end

        clear chanVar
        for b = 1:length(files1)
            % For the purposes of looking for bad channels, it seems most sensible to leave the BDF referenced as it was recorded.
            % If we average-reference, a bad channel's badness is diluted and may spread to other channels.
            % With a single reference channel, it would be ok, as long as that channel is clean.

            if ismember(sessionID{s},monashSessionID)
                bl = blockrange{b}*EEG.srate;
                if bl(1) == 0; bl(1) = 1; end % Second 0 is sample 1
                [ filepath , name , ext ] = fileparts(files1{b});
                EEG = pop_loadbv([filepath,'/'], [name , ext],bl);% needs sample range
                clear  filepath  name   ext
            else
                EEG = pop_biosig(files1{b},'blockrange',round(blockrange{b}));% needs time range
            end

            if EEG.srate>500
                EEG = pop_resample(EEG, 500);
            end
            % Fish out the event triggers and times
            clear trigs stimes
            for i = 1:length(EEG.event)
                if ~isa(EEG.event(i).type,'double') && ~startsWith(EEG.event(i).type,'S')
                    EEG.event(i).type = [];
                    EEG.event(i).type = 0;
                elseif  ~isa(EEG.event(i).type,'double') && startsWith(EEG.event(i).type,'S')
                    EEG.event(i).type = regexp(EEG.event(i).type,'\d*','Match');
                    EEG.event(i).type = str2num(EEG.event(i).type{1});
                end
                trigs(i) = EEG.event(i).type;
                stimes(i) = EEG.event(i).latency;
            end
            %         end

            temp = abs(fft(EEG.data(:,stimes(1):stimes(end))'))'; % FFT amplitude spectrum
            tempF = [0:size(temp,2)-1]*EEG.srate/size(temp,2); % Frequency scale
            chanVar(:,b) = mean(temp(:,find(tempF>speclims(1) & tempF<speclims(2))),2);% ROW of variances

        end

        save(matfile{s},'chanlocs','chanVar')
        end_time = toc;
    end

    close(h)

%% -------------------------------------------------------------------------------------------------
% Check4bdChans
% --------------------------------------------------------------------------------------------------
for s = file_start%:length(sessionID)

   close all;
    if ismember(sessionID{s},monashSessionID)
         chanlocs = load('Monash_chanlocs.mat')
    else
        %chanlocs = readlocs('cap64.loc');
         chanlocs = load('UQ_chanlocs.mat');
    end
    disp(sessionID{s})
    load(matfile{s})
    chanVar = double(chanVar);
    avVar = mean(chanVar,2);
    
    % average variance for each channel across all 16 conditions
    % on a second sweep for a given subject, might want to plot topo again
    % after getting rid of a really bad one (to make it easier to see other
    % bad channels) - so do something like:
%     avVar(61) = avVar(56);  % quick hack - make a real bad chan equal its neighbor
    
    figure;
    topoplot(avVar,chanlocs,'plotchans',[1:64],'electrodes','numbers');
    title(sessionID{s})

    figure; hold on
    plot(chanVar(1:64,[1:3 5:10]))
    title(sessionID{s})
    
    close all
    badchans =    [35]; % These two lines will swap and bad channels you identify with the 'changechans' in the next line
    changechans = [36]; % Must be in same order as badchans.
    chanVar(badchans(1:end),:) = chanVar(changechans(1:end),:);
    avVar = mean(chanVar,2);
    figure;
    topoplot(avVar,chanlocs,'plotchans',[1:64],'electrodes','numbers');
    title(sessionID{s})
    figure; hold on
    plot(chanVar(1:64,:))
    title(sessionID{s})
    
end