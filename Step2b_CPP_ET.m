% Define target codes and sampling rates
targcodes = 101:106;          % Continuous Dots
old_fs = 1024;                % Original sample rate
fs = 500;                     % New sample rate

% Define ERP epoch in sample points
ts = -1 * fs : 2.080 * fs;    % -1000ms to 1880ms (including 200ms padding)
t = ts * 1000 / fs;           % Convert to milliseconds
ts_crop = -0.8 * fs : 1.880 * fs;  % Cropped epoch: -800ms to 1880ms
t_crop = ts_crop * 1000 / fs; % Convert cropped epoch to milliseconds

% Baseline interval and response time
BLint = [-100 0];             % Baseline interval in milliseconds
default_response_time = 1.980; % Coherent dot motion duration (1880ms) + 100ms

% Alpha samples and epoch sizes
Alpha_samps = 52;             % Size of epochs in samples (for alpha TSE)
ERP_samps = length(t_crop);   % Number of samples in ERP epoch
Pupil_samps = ERP_samps;      % Number of samples for pupil data

% Number of channels and filter settings
nchan = 64;                   % Number of channels (64 or 72 if EOGs included)
LPFcutoff = 35;               % Low-pass filter cutoff frequency
HPFcutoff = 0;                % High-pass filter cutoff frequency

% Low-pass and high-pass filter options
LPF = 1;                      % 1 = Apply low-pass filter, 0 = Do not apply
HPF = 0;                      % 1 = Apply high-pass filter, 0 = Do not apply

% Alpha bandpass filter settings
bandlimits = [8 13];          % Alpha bandpass filter range (8-13 Hz)
[H1, G1] = butter(4, 2 * (bandlimits / fs));  % 4th-order Butterworth filter

% Pre-target artifact rejection window
PretargetARwindow = [-0.500, 0]; % Time window for pre-target artifacts (in seconds)

% Channels for artifact rejection
ARchans = 1:64;               % Channels used for artifact rejection
ARchans_for_blinks = 1:64;    % Channels used for blink detection

% Artifact threshold and channel tracking
artifth = 100;                % Artifact threshold
artifchans = [];              % Initialize empty array for channels exceeding threshold

if ismember(sessionID{s},monashSessionID)
    chanlocs = load('Monash_chanlocs.mat');
    chanlocs = chanlocs.chanlocs;  
else
    chanlocs = load('UQ_chanlocs.mat');
    chanlocs = chanlocs.chanlocs;  
end
chanlocs = chanlocs(1:nchan)';

clear files matfiles; k=0;

for n=1:length(blocks)
    k=k+1;
    if ismember(sessionID{s},monashSessionID)
        files{k} = [subjID num2str(blocks(n)) '.vhdr'];
        
    else
        files{k} = [subjID num2str(blocks(n)) '.bdf'];
    end
    matfiles{k} = [ subjID num2str(blocks(n)) '.mat'];
    ET_files{k} = [ subjID num2str(blocks(n)) '.asc'];
    ET_matfiles{k} = [subjID num2str(blocks(n)),'_ET.mat'];
end

rejected_trial_n=[];
artifact_PretargetToTarget_n=[];
artifact_BLintTo100msPostResponse_n=[];
artifact_BLintTo450ms_n=[];
artifact_neg1000_to_0ms_n=[];

erp = [];erp_HPF = []; Alpha = []; Pupil=[]; GAZE_X=[]; GAZE_Y=[]; n=0; artifacts_anywhereInEpoch = 0;
allRT=[]; allrespLR=[]; allTrig=[]; numtr=0;    % note allRT will be in sample points
falsealarm=[];
for f = 1: length(files)
    disp(f)
    filename=[files{f}];
    if ismember(subject_folder{s},monashIds)
        [filepath , name , ext ] = fileparts( filename );
        EEG = pop_loadbv([datafolder,subject_folder{s}, filesep], [name,ext]);
        clear filepath  name  ext
    else
        EEG = pop_biosig(filename, 'blockepoch', 'off','channels',[1:nchan]);
    end   
%%
    if EEG.srate~= fs
    EEG2 = pop_resample(EEG,fs);
    else 
        EEG2 = EEG;
    end
    
    numev = length(EEG.event);
    
    load(matfiles{f}); %DN
    
    trialCond = trialCond+100;
   
    % interpolate bad channels
     EEG.chanlocs = chanlocs;
    if ~isempty(badchans)     
        EEG=eeg_interp(EEG,[badchans],'spherical');
    end
    
    EEG.data = double(EEG.data);
    
     % First LP Filter
    if LPF, EEG = pop_eegfiltnew(EEG, 0, LPFcutoff); end %new FIR filte
    %% --------------------------------------------------------------------
    %% --------------------------------------------------------------------
    
    EEG_HPF = EEG;
    
    % new HPF method
    if HPF, EEG_HPF = pop_eegfiltnew(EEG_HPF, HPFcutoff,0); end
    
    % average-reference the whole continuous data (safe to do this now after interpolation):
    EEG.data = EEG.data - repmat(mean(EEG.data([1:nchan],:),1),[nchan,1]);
    EEG_HPF.data = EEG_HPF.data - repmat(mean(EEG_HPF.data([1:nchan],:),1),[nchan,1]);
    
    %% Sync up the Eyelink data:
    if exist(ET_matfiles{f}, 'file') && exist(ET_files{f}, 'file')%# MB Temp
        delete(ET_matfiles{f})%# MB Temp
    end
    
     if ~exist(ET_matfiles{f}, 'file') %DN: if ET matfile has NOT has been saved previouslty,
        FixEyelinkMessages %then calculate and save it now
     end
    load(ET_matfiles{f}) %DN: load the ET mat file
    %Add an extra 4 rows into the EEG struct - 'TIME'
    %'GAZE_X' 'GAZE_Y' 'AREA'. This will add these as extra channels onto EEG.data
    %So the final channel is the pupil area (i.e. diameter):
    
    if strcmp(subject_folder(s),'C183') && f==8  % C183's block 8 has first trigger missing in the EEG, so make first trigger for Eyetracker the first target existing in the EEG which is "105" for syncing up EEG and eyetracking
        first_event=105;
    end
    
    if strcmp(subject_folder(s),'AD46C') && f==6  % AD46C's block 6 has first few triggers missing in the EEG, so make first trigger for Eyetracker the first target existing in the EEG which is "104" for syncing up EEG and eyetracking
        first_event=104;
    end
    
    EEG = pop_importeyetracker(EEG,ET_matfiles{f},[first_event last_event],[1:4] ,{'TIME' 'GAZE_X' 'GAZE_Y' 'AREA'},0,1,0,0);
    
    
    Pupil_ch=length(EEG.data(:,1)); %Now that the eyelink data is added to the EEG struct Find the channel number for pupil area/diameter
    GAZE_X_ch=length(EEG.data(:,1))-2;
    GAZE_Y_ch=length(EEG.data(:,1))-1;
    
    if EEG.srate~=fs
    EEG = pop_resample(EEG, fs);
    end
    if HPF
        EEG_HPF = pop_resample( EEG_HPF, fs);
    end
        
    % Fish out the event triggers and times
    clear trigs stimes RT
    for i=1:numev
        if ~isa(EEG2.event(i).type,'double') && ~startsWith(EEG2.event(i).type,'S','IgnoreCase',true)&& isnan(str2double(EEG.event(i).type))...
                && ~startsWith(EEG2.event(i).type,'Condition','IgnoreCase',true)
            EEG2.event(i).type = [];
            EEG2.event(i).type = 0;
        else
            EEG2.event(i).type = regexp(EEG2.event(i).type,'\d*','Match');
            EEG2.event(i).type = str2num(EEG2.event(i).type{1});
        end
        
        trigs(i)=EEG2.event(i).type;
        stimes(i)=round(EEG2.event(i).latency);
    end
    targtrigs = [];
    for n=1:length(trigs)
        if any(targcodes(:)==trigs(n))
            targtrigs = [targtrigs n];
        end
    end
    
    if trigs(targtrigs(end))==trialCond(1)
        motion_on = targtrigs(1:end-1); % GL: indices of trigs when motion on. get rid of last trig, it was a repeat
    else
        motion_on = targtrigs;
    end
    
    cohmo_trigs = find(PTBtrig>100); %DN
    rtprelim = 0; %search for response before 200ms
    rtlim=[0.2 2]; %DN: RT must be between 200ms and 2000ms
    
    if length(RespLR)<length(RTs)
        RespLR(length(RTs))=0; %DN: this is just for if they missed the last target the paradigm code won't have recorded "0" for RespLR this makes it 0
    end
    
    %%
    found = 0;
   
    if length(motion_on)<length(trialCond) % There's a trigger missing
        disp('Trigger Missing')
        for n=1:length(motion_on)
            if trigs(motion_on(n))~=trialCond(n)
                found = 1;
                motion_on = [motion_on(1:n-1),NaN,motion_on(n:end)];
            end
        end
        if found==0
            motion_on = [motion_on,NaN];
        end
    elseif length(motion_on)>length(trialCond) % There's an extra trigger
        disp('Extra Trigger')
        for n=1:length(trialCond)
            if trigs(motion_on(n))~=trialCond(n) % e.g. trigger 12 extra, trigs(motion_on(49))==trialCond(48)
                motion_on(n) = [];
            end
        end
    end
    
    for n=1:length(motion_on)
        clear ep ep_HPF ep_alpha ep_art_reject ep_test ep_filt_Alpha_Hz ep_filt_abs_cut ep_pupil ep_GAZE_X ep_GAZE_Y %ep_CSD
        if ~isnan(motion_on(n))
            locktime = stimes(motion_on(n)); % Lock the epoch to coherent motion onset.
            % If they completed the original version of paradigm (i.e. the first 5 ADHD kids only) must calculate RT and accuracy like this:
            try
                if ismember(subject_folder{s},original_paradigm)
                    stimtime = PTBtrigT(cohmo_trigs(n));
                    nextresp = find(RespT>(stimtime-rtprelim) & RespT<stimtime+rtlim(2),1);
                    if ~isempty(nextresp)
                        response_time = (RespT(nextresp) - stimtime)*fs;
                        response_time = floor(response_time); % round it off.
                    else
                        response_time = default_response_time*fs; % there was no response, set response to 1980ms. This is just to define the epoch for artifact rejection
                    end
                    % Else they completed the updated paradigm (i.e. everybody else) can calculate RT like this:
                elseif RespLR(n) %if a response was made:
                    stimtime = PTBtrigT(cohmo_trigs(n));
                    if ~RespT_LeftClick1st(n) && ~RespT_RightClick1st(n)
                        if RespT(n)>(stimtime-rtprelim)
                            response_time = (RespT(n) - stimtime)*fs; % if negative, then false alarm
                            response_time = floor(response_time); % round it off.
                        else %else response was made before the stimtime or at exactly the stimtime
                            response_time = default_response_time*fs; % since response was made before or at stimtime, set response to 1980ms. This is just to define the epoch for artifact rejection
                        end
                    elseif RespT_LeftClick1st(n)
                        if (RespT(n)+RespT_LeftClick1st(n))/2 > (stimtime-rtprelim)
                            response_time =((RespT(n)+RespT_LeftClick1st(n))/2 - stimtime)*fs; %RT is average of when the two buttons were pressed
                            response_time = floor(response_time); % round it off.
                        else % else response was made before the stimtime or at exactly the stimtime
                            response_time = default_response_time*fs; % since response was made before or at stimtime, set response to 1980ms. This is just to define the epoch for artifact rejection
                        end
                    elseif RespT_RightClick1st(n)
                        if (RespT(n)+RespT_RightClick1st(n))/2 > (stimtime-rtprelim)
                            response_time =((RespT(n)+RespT_RightClick1st(n))/2 - stimtime)*fs; %RT isaverage of when the two buttons were pressed
                            response_time = floor(response_time); % round it off.
                        else %else response was made before the stimtime or at exactly the stimtime
                            response_time = default_response_time*fs; % since response was made before or at stimtime, set response to 1980ms. This is just to define the epoch for artifact rejection
                        end
                    end
                else % there was no response
                    stimtime = PTBtrigT(cohmo_trigs(n));
                    response_time = default_response_time*fs; % there was no response, set response to 1980ms. This is just to define the epoch for artifact rejection
                end
            catch
                disp('EEG ended too soon') % I had a few trials where EEG was cutoff too soon...
                response_time = default_response_time*fs;
            end
            try
                ep = EEG.data(1:nchan,locktime+ts);   % chop out an epoch
                %ep_CSD = CSD(ep, G_CSD, H_CSD); % CSD epoch
                ep_HPF = EEG_HPF.data(1:nchan,locktime+ts);
            catch
                disp('EEG ended too soon')
                %%%%%%%%%
                numtr = numtr+1;
                rejected_trial_n(numtr)=1;
                %%%%%%%%%
                allTrig(numtr) = 0;
                allblock_count(numtr) = f;
                allrespLR(numtr) = NaN;
                allRT(numtr) = NaN;
                falsealarm(numtr)=NaN;
                
                erp(:,:,numtr) = zeros(nchan,ERP_samps);
                erp_HPF(:,:,numtr) = zeros(nchan,ERP_samps);
                Alpha(:,:,numtr) = zeros(nchan,Alpha_samps);
                
                Pupil(:,numtr) = zeros(1,Pupil_samps);
                GAZE_X(:,numtr) = zeros(1,Pupil_samps);
                GAZE_Y(:,numtr) = zeros(1,Pupil_samps);
                
                %                         keyboard
                continue;
            end
                      
            
            try
                ep_pupil = EEG.data(Pupil_ch,locktime+ts);   % chop out an epoch of pupil diameter
                ep_GAZE_X = EEG.data(GAZE_X_ch,locktime+ts); % chop out an epoch of GAZE_X
                ep_GAZE_Y = EEG.data(GAZE_Y_ch,locktime+ts); % chop out an epoch of GAZE_Y
            catch
                disp('Pupil Diameter data ended too soon')      
                numtr = numtr+1;
                rejected_trial_n(numtr)=1;
                allTrig(numtr) = 0;
                allrespLR(numtr) =NaN;
                allRT(numtr) = NaN;
                allblock_count(numtr) = f;
                 falsealarm(numtr)=NaN;
                Pupil(:,numtr) = zeros(1,Pupil_samps);
                GAZE_X(:,numtr) = zeros(1,Pupil_samps);
                GAZE_Y(:,numtr) = zeros(1,Pupil_samps);
                
                keyboard
                continue;
            end
            
            BLamp =mean(ep(:,find(t>BLint(1) & t<BLint(2))),2); % record baseline amplitude (t<0) for each channel,
            ep = ep - repmat(BLamp,[1,length(t)]); % baseline correction
            
            BLamp = mean(ep_HPF(:,find(t>BLint(1) & t<BLint(2))),2);
            ep_HPF = ep_HPF - repmat(BLamp,[1,length(t)]); % baseline correction
            
            
            ep_test = [find(ts==-0.8*fs):find(ts==(0*fs))];
            if isempty(ep_test)
                disp('Empty epoch for art rejection')
                keyboard
            end
            ep_test = [find(t>BLint(1) & t<floor(((response_time*1000/fs)+100)))];
            if isempty(ep_test)
                disp('Empty epoch for art rejection2')
                keyboard
            end

            numtr = numtr+1;
            rejected_trial_n(numtr)=0;

            allblock_count(numtr) = f;
            
            artifchans_thistrial = ARchans(find(max(abs(ep_HPF(ARchans,find(t<0))),[],2)>artifth | max(abs(ep_HPF(ARchans,find(t>BLint(1) & t<floor(((response_time*1000/fs)+100))))),[],2)>artifth));
            
            artifchans_blinks_thistrial = ARchans(find(max(abs(ep_HPF(ARchans_for_blinks,find(t<0))),[],2)>artifth));
            
            artifchans_blinks_thistrial(find(ismember(artifchans_blinks_thistrial,artifchans_thistrial))) = [];
            artifchans_thistrial = [artifchans_thistrial,artifchans_blinks_thistrial(find(~ismember(artifchans_blinks_thistrial,artifchans_thistrial)))];
            artifchans = [artifchans artifchans_thistrial];
            
            artifchans_PretargetToTarget_thistrial = ARchans(find(max(abs(ep_HPF(ARchans,find(ts==PretargetARwindow(1)*fs):find(ts==0))),[],2)>artifth));  % pre-target artifact rejection from -500-0ms only [find(ts==-.500*fs) gives you the point in samples -500ms before the target]
            artifchans_BLintTo100msPostResponse_thistrial = ARchans(find(max(abs(ep_HPF(ARchans,find(t>BLint(1) & t<floor(((response_time*1000/fs)+100))))),[],2)>artifth)); %Baseling until 100ms after response.
            
            artifchans_BLintTo450ms_thistrial = ARchans(find(max(abs(ep_HPF(ARchans,find(ts==-0.1*fs):find(ts==0.45*fs))),[],2)>artifth));  % artifact rejection from -100 to 500ms only
            
            
            if ~isempty(artifchans_thistrial)
                artifacts_anywhereInEpoch = artifacts_anywhereInEpoch+1;
            end   % artifact rejection (threshold test)
            
            if artifchans_PretargetToTarget_thistrial
                artifact_PretargetToTarget_n(numtr)=1;
            else
                artifact_PretargetToTarget_n(numtr)=0;
            end
            
            if artifchans_BLintTo100msPostResponse_thistrial
                artifact_BLintTo100msPostResponse_n(numtr)=1;
            else
                artifact_BLintTo100msPostResponse_n(numtr)=0;
            end
            
            if artifchans_BLintTo450ms_thistrial
                artifact_BLintTo450ms_n(numtr)=1;
            else
                artifact_BLintTo450ms_n(numtr)=0;
            end
            

            ep_pupil = double(ep_pupil);
            ep_GAZE_X = double(ep_GAZE_X);
            ep_GAZE_Y = double(ep_GAZE_Y);
            
            ep = double(ep); % filtfilt needs doubles
       
            %%  Alpha::
            for q = 1:size(ep,1) % alpha filter
                ep_filt_Alpha_Hz(q,:) = filtfilt(H1,G1,ep(q,:));
            end
            
            % rectifying the data and chopping off ends
            ep_filt_abs_cut = abs(ep_filt_Alpha_Hz(:,find(ts==ts_crop(1)):find(ts==ts_crop(end)))); % 64x701
            % Smoothing. This goes from 1:700, leaving out the final sample, 0ms.
            alpha_temp = []; Alpha_smooth_time = []; Alpha_smooth_sample = [];
            for q = 1:size(ep_filt_abs_cut,1)
                counter = 1;
                for windowlock = 26:25:size(ep_filt_abs_cut,2)-25 % 1,26,51,etc. 26 boundaries = 1:50, 51 boundaries = 26:75, 676 boundaries = 651:
                    alpha_temp(q,counter) = mean(ep_filt_abs_cut(q,windowlock-25:windowlock+24));
                    Alpha_smooth_time(counter) = t_crop(windowlock);
                    Alpha_smooth_sample(counter) = ts_crop(windowlock);
                    counter = counter+1;
                end
            end
                        % 
                        % figure
                        % plot(ep_filt_abs_cut(54,:))
                        % figure
                        % plot(Alpha_smooth_time,alpha_temp(54,:))
                        % keyboard
                        % 
                        % figure, hold on, for i = 1:64, plot(ep(i,find(ts==ts_crop(1)):find(ts==ts_crop(end))),'b'), end; keyboard
                        % 
            
            erp(:,:,numtr) = ep(:,find(ts==ts_crop(1)):find(ts==ts_crop(end)));
            %erp_CSD(:,:,numtr) = ep_CSD(:,find(ts==ts_crop(1)):find(ts==ts_crop(end)));
            erp_HPF(:,:,numtr) = ep_HPF(:,find(ts==ts_crop(1)):find(ts==ts_crop(end)));
            Pupil(:,numtr)= ep_pupil(find(ts==ts_crop(1)):find(ts==ts_crop(end)));
            GAZE_X(:,numtr)= ep_GAZE_X(find(ts==ts_crop(1)):find(ts==ts_crop(end)));
            GAZE_Y(:,numtr)= ep_GAZE_Y(find(ts==ts_crop(1)):find(ts==ts_crop(end)));
            Alpha(:,:,numtr) = alpha_temp;
            allTrig(numtr) = trigs(motion_on(n));
            
            try % get reaction time data for further analysis
               % If they completed the original version of paradigm (i.e. the first 5 ADHD kids only) must calculate RT and accuracy like this:
                if ismember(subject_folder{s},original_paradigm)
                    stimtime = PTBtrigT(cohmo_trigs(n));
                    nextresp = find(RespT>(stimtime-rtprelim) & RespT<stimtime+rtlim(2),1);
                    if ~isempty(nextresp)
                        allrespLR(numtr) = 1;
                        allRT(numtr) = round((RespT(nextresp) - stimtime)*fs);
                        falsealarm(numtr)=0;
                    else %there was no response
                        allrespLR(numtr) = 0;
                        allRT(numtr) = NaN;
                        falsealarm(numtr)=NaN;
                    end
                   % Else they completed the updated paradigm (i.e. everybody else) can calculate RT more accuratly like this:
                elseif RespLR(n) %if a response was made:
                    stimtime = PTBtrigT(cohmo_trigs(n));
                    allrespLR(numtr) = 1; %correct response
                    if ~RespT_LeftClick1st(n) && ~RespT_RightClick1st(n) %they clicked both buttons at exactly the same time
                        if RespT(n)>(stimtime-rtprelim)
                            allRT(numtr) = round((RespT(n) - stimtime)*fs);
                            falsealarm(numtr)=0;
                        else 
                            falsealarm(numtr) = round((RespT(n) - stimtime)*fs);
                        end
                    elseif RespT_LeftClick1st(n)
                        if (RespT(n)+RespT_LeftClick1st(n))/2 > (stimtime-rtprelim)
                            allRT(numtr) =round(((RespT(n)+RespT_LeftClick1st(n))/2 - stimtime)*fs); %RT is average of when the two buttons were pressed
                            falsealarm(numtr)=0;
                        else
                            falsealarm(numtr) = round(((RespT(n)+RespT_LeftClick1st(n))/2 - stimtime)*fs);
                        end
                    elseif RespT_RightClick1st(n)
                        if (RespT(n)+RespT_RightClick1st(n))/2 > (stimtime-rtprelim)
                            allRT(numtr) =round(((RespT(n)+RespT_RightClick1st(n))/2 - stimtime)*fs); %RT isaverage of when the two buttons were pressed
                            falsealarm(numtr)=0;
                        else
                            falsealarm(numtr) = round(((RespT(n)+RespT_RightClick1st(n))/2 - stimtime)*fs);
                        end
                    end
                else % there was no response
                    allrespLR(numtr) = 0;
                    allRT(numtr) = NaN;
                    falsealarm(numtr)=NaN;
                end
            catch
                allrespLR(numtr) = 0;
                allRT(numtr) = NaN;
                falsealarm(numtr)=NaN;
            end
        else
            numtr = numtr+1;
            rejected_trial_n(numtr)=1;
            allTrig(numtr) = 0;
            allblock_count(numtr) = f;
            allrespLR(numtr) = 0;
            allRT(numtr) = NaN;
            falsealarm(numtr)=NaN;
            erp(:,:,numtr) = zeros(nchan,ERP_samps);
            erp_HPF(:,:,numtr) = zeros(nchan,ERP_samps);
            Alpha(:,:,numtr) = zeros(nchan,Alpha_samps);
            Pupil(:,numtr) = zeros(1,Pupil_samps);
            GAZE_X(:,numtr) = zeros(1,Pupil_samps);
            GAZE_Y(:,numtr) = zeros(1,Pupil_samps);
        end
    end
end

Alpha = Alpha(1:nchan,:,:);
erp = erp(1:nchan,:,:);
erp_HPF = erp_HPF(1:nchan,:,:);
Pupil=Pupil(:,:);
GAZE_X=GAZE_X(:,:);
GAZE_Y=GAZE_Y(:,:);
% figure;
% hist(artifchans,[1:nchan]); title([subject_folder{s} ': ' num2str(artifacts_anywhereInEpoch) ' artifacts = ',num2str(round(100*(artifacts_anywhereInEpoch/length(allRT)))),'%']) % s from runafew
% disp([subject_folder{s},' number of trials: ',num2str(length(allRT))])
% if length(allRT)~=size(Alpha,3)
%     disp(['WTF ',subject_folder{s},' number of trials: ',num2str(length(allRT)),' not same as Alpha'])
% end
save([pathOut, num2str(s), '_' subject_folder{s},'_',num2str(bandlimits(1,1)),'_to_',num2str(bandlimits(1,2)), ...
    'Hz_neg',num2str(abs(t_crop(1))),'_to_',num2str(t_crop(end)),'_',num2str(length(ARchans)),'ARchans',...
    '_',num2str(LPFcutoff),'HzLPF_point',strrep(num2str(HPFcutoff),'0.',''),'HzHPF_ET_',partGroup],...
    'Alpha','erp','erp_HPF','Pupil','GAZE_Y','GAZE_X','allRT','allrespLR','allTrig','allblock_count', ...
    'artifchans','t_crop','Alpha_smooth_time','Alpha_smooth_sample',...
    'artifact_PretargetToTarget_n','artifact_BLintTo100msPostResponse_n',...
    'rejected_trial_n','artifact_BLintTo450ms_n','falsealarm')%,'erp_CSD'
close all
return;
