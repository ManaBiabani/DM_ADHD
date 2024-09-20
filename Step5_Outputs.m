clear all; close all; clc

% --------------------------------------------------------------------------------------------------
% Setup paths to code, data, and functions
% --------------------------------------------------------------------------------------------------
% Change directory to the relevant folder containing the decision-making data
cd('/Users/manab/GoogleDrive_Gmail/DecisionMaking/')

% Add the necessary EEGLAB toolbox and custom functions to the MATLAB path
addpath(genpath('/Users/manab/Desktop/Functions/eeglab2021.0/')); % EEGLAB functions
addpath(genpath('/Users/manab/GoogleDrive_Gmail/Code/functionsInUse')); % Custom functions

% Initialize EEGLAB
eeglab;

% Define output path for data and scripts
pathOut = '/Users/manab/Desktop/ADHDKids/Analysis2023/dataScriptsResults/';

% --------------------------------------------------------------------------------------------------
% Load necessary data files
% --------------------------------------------------------------------------------------------------
% Load the behavioral data and common channel information
load([pathOut, 'DM_beh.mat']);       % Load behavioral data
load('commonChans.mat');             % Load common channels data
load('common_chanlocs.mat');         % Load common channel locations data

% --------------------------------------------------------------------------------------------------
% Channel names for EEG analysis
% --------------------------------------------------------------------------------------------------
% Define relevant EEG channels for left/right hemisphere and N2c/ipsi-con channels
ch_LR_name = {'P7','P8'};              % Lateralized response channels (Left: P7, Right: P8)
ch_N2c_name = {'P8','P7'};             % N2c channels (opposite hemisphere)
ch_for_ipsicon_name(1,:) = {'P8','P7'}; % Ipsilateral configuration (first set: P8 left, P7 right)
ch_for_ipsicon_name(2,:) = {'P7','P8'}; % Ipsilateral configuration (second set: P7 left, P8 right)
ch_l_name = 'P7';                      % Left hemisphere channel
ch_r_name = 'P8';                      % Right hemisphere channel

% Sampling frequency and time window setup for ERP epoch
fs = 500;                               % Sampling frequency (Hz)
ts = -0.8*fs:1.880*fs;                  % Time range in sample points (-800 ms to 1880 ms)
t = ts * 1000 / fs;                     % Convert to milliseconds

% Cropped time window setup
ts_crop = -0.8*fs:1.880*fs;             % Cropped time window in sample points
t_crop = ts_crop * 1000 / fs;           % Convert to milliseconds

% Response-locked epoch time window
trs = [-0.800*fs:fs*0.100];             % Response-locked epoch (-800 ms to +100 ms)
tr = trs * 1000 / fs;                   % Convert to milliseconds

% --------------------------------------------------------------------------------------------------
% Subsample the data based on site (Monash vs UQ)
% --------------------------------------------------------------------------------------------------
% Check if each subject belongs to Monash or UQ datasets
a = zeros(1, length(groupAll));          % Initialize array for site labels (1: Monash, 0: UQ)
for s = 1:length(groupAll)
    if ismember(subject_folder{s}, monashIds)
        a(s) = 1;                        % Assign 1 for Monash
    else
        a(s) = 0;                        % Assign 0 for UQ
    end
end

% Separate UQ and Monash data based on the site array
UQ_data = find(a == 0);                  % Indices for UQ data
Monash_data = find(a == 1);              % Indices for Monash data

% --------------------------------------------------------------------------------------------------
% Exclude specific subjects (due to missing/invalid data)
% --------------------------------------------------------------------------------------------------
% List of subject IDs to be excluded
excIdsInd = find(ismember(subject_folder, {'C14', 'C21', 'C507', 'C88', 'AD91C', 'AD19C'})); 

% Display the subject folders being excluded
disp('Excluding the following subject folders:');
disp(subject_folder(excIdsInd));

% --------------------------------------------------------------------------------------------------
% Create new subject folder excluding the invalid subjects
% --------------------------------------------------------------------------------------------------
newSubject_folder = subject_folder;      % Copy original subject folders
newSubject_folder(excIdsInd) = [];       % Remove excluded subjects from the new list

% --------------------------------------------------------------------------------------------------
% Create site and group vectors excluding the invalid subjects
% --------------------------------------------------------------------------------------------------
% Initialize site vector (1 = Monash, 2 = UQ)
site = ones(length(groupAll), 1);
site(UQ_data) = 2;                       % Assign 2 for UQ data
site(excIdsInd) = [];                    % Remove excluded subjects from site vector

% Remove excluded subjects from the group vector as well
groupAll(excIdsInd) = [];

% Set number of channels
numch = 58;                              % Number of channels to be used in the analysis
% --------------------------------------------------------------------------------------------------
% Define plotting parameters
% --------------------------------------------------------------------------------------------------
Patchside = {'Left Target ', 'Right Target '};      % Labels for left/right target patches
ADHD_Control = {'ADHD ', 'Control '};               % Labels for ADHD and Control groups
colors = {'b', 'r', 'g', 'm', 'c'};                 % Colors for plots (blue, red, green, magenta, cyan)
line_styles = {'-', '--'};                          % Line styles for plots
line_styles2 = {'--', '--', '-', '-'};              % Additional line styles
colors_coh = {'r', 'b'};                            % Colors for coherence plots (red, blue)
line_styles3 = {'-', '--', '-', '--'};              % Another set of line styles
linewidths = [1.5, 1.5, 1.5, 1.5, 1.3, 1.3, 1.3, 1.3];  % Line widths for plots
linewidths2 = [1.5, 1.5, 1.3, 1.3];                 % Additional line widths
colors3 = {'b', 'b', 'r', 'r'};                     % Colors for left/right target (blue for left, red for right)

% --------------------------------------------------------------------------------------------------
% New channel selection for left and right hemisphere electrodes
% --------------------------------------------------------------------------------------------------
leftHemChan = {'PO8', 'PO4'};  % Channels for the left hemisphere (posterior)
rightHemChan = {'PO7', 'PO3'}; % Channels for the right hemisphere (posterior)

% Find the electrode indices based on the channel labels in 'chanlocs'
LH_elec = zeros(1, length(leftHemChan));  % Preallocate array for left hemisphere electrode indices
RH_elec = zeros(1, length(rightHemChan)); % Preallocate array for right hemisphere electrode indices

for el = 1:length(leftHemChan)
    LH_elec(el) = find(strcmp({chanlocs.labels}, leftHemChan{el}));  % Find index for each left hemisphere channel
end

for el = 1:length(rightHemChan)
    RH_elec(el) = find(strcmp({chanlocs.labels}, rightHemChan{el})); % Find index for each right hemisphere channel
end

% --------------------------------------------------------------------------------------------------
% Prepare for Region of Interest (ROI) data extraction for each participant
% --------------------------------------------------------------------------------------------------
% Create a 3D array (participants x electrodes x hemispheres) for left and right hemisphere channels
ROIs_LH_RH = zeros(length(groupAll), length(rightHemChan), 2);  % Preallocate ROI matrix
ROIs_LH_RH(:, :, 1) = repmat(LH_elec, length(groupAll), 1);     % Assign left hemisphere electrode indices
ROIs_LH_RH(:, :, 2) = repmat(RH_elec, length(groupAll), 1);     % Assign right hemisphere electrode indices

% --------------------------------------------------------------------------------------------------
% Extract Pre-target Alpha per participant
% --------------------------------------------------------------------------------------------------
% Choose dataset based on the specified group ('all', 'Monash', 'UQ')
whichData = 'all';  % Options: 'all', 'Monash', 'UQ'

% Select the Alpha data for pre-target condition
dtt = Alpha_cond_preAll;                % All pre-target Alpha data
dtt(excIdsInd, :, :, :, :) = [];        % Exclude invalid subjects (based on excIdsInd)

% Initialize variables to store data and group information
clearvars dt groupAll_dt

% Subset the data based on the specified group
if strcmp(whichData, 'all')
    dt = dtt;                           % Use all data
    groupAll_dt = groupAll;              % Use all group labels
elseif strcmp(whichData, 'Monash')
    dt = dtt(find(site == 1), :, :, :, :);   % Use only Monash data
    groupAll_dt = groupAll(find(site == 1)); % Use corresponding Monash group labels
elseif strcmp(whichData, 'UQ')
    dt = dtt(find(site == 2), :, :, :, :);   % Use only UQ data
    groupAll_dt = groupAll(find(site == 2)); % Use corresponding UQ group labels
end

% --------------------------------------------------------------------------------------------------
% Compute the mean Pre-target Alpha power for each participant across specified ROIs
% --------------------------------------------------------------------------------------------------
% Iterate through each participant to calculate mean Pre-target Alpha power in both hemispheres
PreAlpha_mean_allWindow = zeros(size(dt, 1), 2, size(dt, 3));  % Preallocate result matrix
BL = [-700 -400];

for s = 1:size(dt, 1)  % Loop over participants
    for hemi = 1:2      % Loop over hemispheres (1 = left, 2 = right)
        % Compute the mean Alpha power across the selected ROIs (left/right hemisphere)
        PreAlpha_mean_allWindow(s, hemi, :) = squeeze(mean(mean(dt(s, ROIs_LH_RH(s, :, hemi), :, hemi, :), 2), 5));
    end
end

a = [];
b = [];
for idx = 1:79
    for hemi = 1:2

a(idx,hemi,:) = squeeze(PreAlpha_mean_allWindow(idx,hemi,:))- nanmean(squeeze(PreAlpha_mean_allWindow(idx,hemi,find(Alpha_smooth_time==BL(1)):find(Alpha_smooth_time==BL(end)))));

    end

end

% Now, 'PreAlpha_mean_allWindow' contains the average Pre-target Alpha power for each participant,
% computed separately for the left and right hemispheres.
PreAlpha_collapsed = nanmean(nanmean(a(:,:,find(Alpha_smooth_time==BL(1)):find(Alpha_smooth_time==BL(end))),3),2);

% --------------------------------------------------------------------------------------------------
% Extract N2c Amplitude and Latency Based on Each Participant's N2c
% --------------------------------------------------------------------------------------------------
% Load the data, removing excluded participants
dtt = N2_condAll;
dtt(excIdsInd, :, :, :, :) = [];
clearvars dt groupAll_dt

% Select which data to use (all, Monash, or UQ)
if strcmp(whichData, 'all')
    dt = dtt;
    groupAll_dt = groupAll;
elseif strcmp(whichData, 'Monash')
    dt = dtt(find(site == 1), :, :, :, :);
    groupAll_dt = groupAll(find(site == 1));
elseif strcmp(whichData, 'UQ')
    dt = dtt(find(site == 2), :, :, :, :);
    groupAll_dt = groupAll(find(site == 2));
end

% Define parameters
window = 25;  % Time window around the peak (25 samples on each side, equivalent to 50 ms)
group_path = {'ADHD', 'Control'};

% Define channels for N2c based on the left and right hemisphere
ch_N2c(1, 1) = find(strcmp(lower(ch_N2c_name{1}), lower({chanlocs.labels})));
ch_N2c(2, 1) = find(strcmp(lower(ch_N2c_name{2}), lower({chanlocs.labels})));

% Loop through each participant and extract N2c amplitude and latency
for s = 1:size(groupAll_dt, 2)
    for TargetSide = 1:2
        % Calculate the average N2c for each target side
        avN2c = squeeze(mean(mean(mean(mean(dt(s, ch_N2c(TargetSide, :), :, TargetSide, :), 5), 4), 2), 1)); % Average over channels and trials
        
        % Find the peak amplitude and its index within a 150-400 ms window
        avN2c_peak_amp = min(avN2c(find(t == 150):find(t == 400)));
        avN2c_peak_amp_index(s, TargetSide) = find(avN2c == avN2c_peak_amp); % Peak latency index
        avN2c_peak_amp_index_t(s, TargetSide) = t_crop(avN2c_peak_amp_index(s, TargetSide)); % Convert to time
        
        % Extract the N2c amplitude within a window around the peak
        max_peak_N2c(s, TargetSide) = squeeze(mean(mean(mean(dt(s, ch_N2c(TargetSide, :), ...
            avN2c_peak_amp_index(s) - window:avN2c_peak_amp_index(s) + window, TargetSide, :), 2), 3), 5));
    end
    
    % Save N2c peak amplitude and latency for each participant
    avN2c_ParticipantLevel_peak_amp_index_s = avN2c_peak_amp_index(s, :);
    % Uncomment the line below if you want to save individual participant data
    % save([pathOut, group_path{group}, filesep, newSubject_folder{s}, filesep, 'avN2c_ParticipantLevel_peak_amp_index.mat'], 'avN2c_ParticipantLevel_peak_amp_index_s');
end

% Collect results across participants for further analysis
N2c_amp_ByTargetSide_ParticipantLevel = max_peak_N2c;  % (LeftTargetN2c, RightTargetN2c)
N2c_indiv = mean(max_peak_N2c, 2);  % Averaged N2c across target sides

% N2c Latency
N2c_latency_ByTargetSide = avN2c_peak_amp_index_t;  % (LeftTargetN2c_latency, RightTargetN2c_latency)
N2cLatency = mean(avN2c_peak_amp_index_t, 2);  % Averaged N2c latency across target sides

% --------------------------------------------------------------------------------------------------
% Original CPP onset detection method
% --------------------------------------------------------------------------------------------------
Cpp_onset_original =  nanmean(CPP_patch_onsetsAll,2);
Cpp_onset_original(excIdsInd)= [];

% --------------------------------------------------------------------------------------------------
% Response-Locked CPP Slope
% --------------------------------------------------------------------------------------------------

% Define which dataset to analyze: 'all', 'Monash', or 'UQ'
whichData = 'all'; 

ch_CPP_name = {'Pz';'Poz'};

labels = {'ADHD Left target','ADHD Right Target','Control Left Target','Control Right Target'};
colors4 = {'b' 'r' 'b' 'r'};

ch_CPP(1) = find(strcmp(lower(ch_CPP_name{1}), lower({chanlocs.labels})));
ch_CPP(2) = find(strcmp(lower(ch_CPP_name{2}), lower({chanlocs.labels})));

close all;

% Labels for plotting
labels = {'ADHD Left Target', 'ADHD Right Target', 'Control Left Target', 'Control Right Target'};
colors4 = {'b', 'r', 'b', 'r'};

% Load the response-locked ERP data
dtt = ERPr_condAll;
dtt(excIdsInd, :, :, :, :) = []; % Remove excluded subjects

% Select dataset based on the site
clearvars dt groupAll_dt
if strcmp(whichData, 'all')
    dt = dtt;
    groupAll_dt = groupAll;
elseif strcmp(whichData, 'Monash')
    dt = dtt(find(site == 1), :, :, :, :);
    groupAll_dt = groupAll(find(site == 1));
elseif strcmp(whichData, 'UQ')
    dt = dtt(find(site == 2), :, :, :, :);
    groupAll_dt = groupAll(find(site == 2));
end

% Define the time frame for calculating the CPP slope (in ms, relative to response)
slope_timeframe = [-450, 50]; % Time window from -450 ms to 50 ms relative to response

% Initialize matrix to store CPP slopes for each subject and target side
CPP_slope = zeros(length(groupAll), 2); 

% Loop through each subject to calculate the response-locked CPP slope
for s = 1:length(groupAll_dt)
    for targetside = 1:2
        % Average ERP data across trials and channels for the current subject and target side
        avERPr_temp = squeeze(nanmean(dt(s, :, :, targetside, :), 5)); % (subject, channels, timepoints)
        
        % Extract the CPP channel (Pz, typically) and average across it
        avERPr_plot = squeeze(mean(avERPr_temp(ch_CPP, :), 1)); % Average across the CPP channel
        
        % Fit a straight line to the CPP waveform in the specified time window (-450 to 50 ms)
        coef = polyfit(tr(tr > slope_timeframe(1) & tr < slope_timeframe(2)), ...
                       avERPr_plot(tr > slope_timeframe(1) & tr < slope_timeframe(2)), 1);
        
        % The slope of the fitted line is the first coefficient (CPP build-up rate)
        CPP_slope(s, targetside) = coef(1);
    end
end

% Collapse across left and right target sides by averaging the slopes
CPP_slopeAll = mean(CPP_slope, 2);

% --------------------------------------------------------------------------------------------------
% Extract Response-Locked CPP Amplitude
% --------------------------------------------------------------------------------------------------

% Define which dataset to analyze: 'all', 'Monash', or 'UQ'
whichData = 'all'; 

close all;

% Labels for the plot
labels = {'ADHD Left Target', 'ADHD Right Target', 'Control Left Target', 'Control Right Target'};
colors4 = {'b', 'r', 'b', 'r'}; % Blue for ADHD, Red for Control, separated by target side

% Load the response-locked ERP data
dtt = ERPr_condAll;
dtt(excIdsInd, :, :, :, :) = []; % Remove excluded subjects

% Select dataset based on the site
clearvars dt groupAll_dt
if strcmp(whichData, 'all')
    dt = dtt;
    groupAll_dt = groupAll;
elseif strcmp(whichData, 'Monash')
    dt = dtt(find(site == 1), :, :, :, :);
    groupAll_dt = groupAll(find(site == 1));
elseif strcmp(whichData, 'UQ')
    dt = dtt(find(site == 2), :, :, :, :);
    groupAll_dt = groupAll(find(site == 2));
end

%% Extract Amplitude

% Initialize matrices to store CPP amplitudes for ADHD and Control groups
CPP_amp_ADHD = zeros(length(find(groupAll == 1)), 2); % ADHD, two target sides
CPP_amp_Control = zeros(length(find(groupAll == 2)), 2); % Control, two target sides

% Loop over both target sides to compute the CPP amplitudes
for targetside = 1:2
    % For ADHD group
    avERPr_temp = squeeze(nanmean(dt(find(groupAll == 1), :, :, targetside, :), 5)); % Average across subjects (ADHD), channels, samples
    CPP_amp_ADHD(:, targetside) = squeeze(nanmean(avERPr_temp(:, ch_CPP, 400), 2)); % Extract CPP amplitude at timepoint 400 ms

    % For Control group
    avERPr_temp = squeeze(nanmean(dt(find(groupAll == 2), :, :, targetside, :), 5)); % Average across subjects (Control), channels, samples
    CPP_amp_Control(:, targetside) = squeeze(nanmean(avERPr_temp(:, ch_CPP, 400), 2)); % Extract CPP amplitude at timepoint 400 ms
end


% Average CPP amplitudes across both target sides for each group
CPP_Amplitude_ADHD = mean([CPP_amp_ADHD(:, 1), CPP_amp_ADHD(:, 2)], 2); % ADHD, collapsed across target sides
CPP_Amplitude_Control = mean([CPP_amp_Control(:, 1), CPP_amp_Control(:, 2)], 2); % Control, collapsed across target sides

% Combine ADHD and Control CPP amplitudes into one array
CPP_Amplitude = [CPP_Amplitude_ADHD; CPP_Amplitude_Control];

% --------------------------------------------------------------------------------------------------
% Extract Resp-Locked CNV Amplitude and Slope 
% --------------------------------------------------------------------------------------------------

% Specify the frontal channel
ch_front_name = 'fz';  % Frontal channel name

% Choose the dataset: 'all', 'Monash', or 'UQ'
whichData = 'all'; 

% Clear previous data variables
dt = [];
dtt = [];

% Load response-locked ERP data and exclude participants
dtt = ERPr_condAll;
dtt(excIdsInd, :, :, :, :) = [];  % Exclude excluded subjects

% Select dataset based on the chosen site (all, Monash, or UQ)
clearvars dt groupAll_dt
if strcmp(whichData, 'all')
    dt = dtt;
    groupAll_dt = groupAll;
elseif strcmp(whichData, 'Monash')
    dt = dtt(find(site == 1), :, :, :, :);
    groupAll_dt = groupAll(find(site == 1));
elseif strcmp(whichData, 'UQ')
    dt = dtt(find(site == 2), :, :, :, :);
    groupAll_dt = groupAll(find(site == 2));
end

% Labels for ADHD and Control groups
labels = {'ADHD', 'Control'};

% Find the index of the frontal channel (Fz)
ch_front = find(strcmpi(ch_front_name, {chanlocs.labels}));

% Time vector for response-locked analysis
tt = tr(find(tr == -800):find(tr == 100));  % Time from -800ms to 100ms

%% Extract CNV Amplitude
% Frontal negativity amplitude averaged between -20ms and 20ms
frontalNeg_Amp = squeeze(nanmean(nanmean(dt(:, ch_front, [find(tt == -20):find(tt == 20)], :), 4), 3)); 
% Averaging over subjects, samples (-20ms to 20ms), and conditions

%% Extract CNV Slope
slope_timeframe = [-70, 0];  % Time window for slope extraction (e.g., from -70ms to 0ms)

% Loop through each subject to calculate the slope of CNV
for s = 1:size(dt, 1)
    % Average ERP over conditions for the current subject
    avERPr_plot = squeeze(nanmean(dt(s, ch_front, :, :), 4));  % Average across conditions
    
    % Fit a linear regression line to the CNV waveform within the specified time window
    coef = polyfit(tr(tr > slope_timeframe(1) & tr < slope_timeframe(2)), avERPr_plot(tr > slope_timeframe(1) & tr < slope_timeframe(2)), 1);
    
    % Extract the slope (first coefficient from polyfit)
    FrontalNeg_slope(s) = coef(1);
end

% --------------------------------------------------------------------------------------------------
% Extract Behaviour
% --------------------------------------------------------------------------------------------------
load([pathOut,'falsealarm.mat'])

RT= RTAll;
RT(excIdsInd)=[];

hitrate = Valid_RT_TrialsAll ./trialCountAll*100;
hitrate(excIdsInd)=[];

for idx = 1:length(subject_folder)
    a = [];
    a = [false_alarmAll{idx,1,1}';false_alarmAll{idx,1,1}';false_alarmAll{idx,1,2}';false_alarmAll{idx,1,3}';...
        false_alarmAll{idx,2,1}';false_alarmAll{idx,2,2}';false_alarmAll{idx,2,3}'];
    falseAlaramCount(idx) = length(find(a~=0));
    % if it is not a valid trial nor a false alarm it is a missed trial
    misses(idx) = trialCountAll(idx)-Valid_RT_TrialsAll(idx)-falseAlaramCount(idx);
end
misses(excIdsInd)=[];

subject_folder(excIdsInd) = [];
%% Make participant level matrix for export into SPSS or R
% Assigning values to participant_level starting from column 1
participant_level(:,1)  = groupAll_dt';           % Column 1: Group
participant_level(:,2)  = site;                   % Column 2: Site
participant_level(:,3)  = RT';                    % Column 3: Reaction Time
participant_level(:,4)  = hitrate';               % Column 4: Hit Rate
participant_level(:,5)  = misses';                % Column 5: Misses
participant_level(:,6)  = PreAlpha_collapsed;     % Column 6: Pre-target Alpha Power
participant_level(:,7)  = N2cLatency;             % Column 7: N2c Latency
participant_level(:,8)  = N2c_indiv;              % Column 8: N2c Amplitude (Individual level)
participant_level(:,9)  = Cpp_onset_original;     % Column 9: CPP Onset
participant_level(:,10) = CPP_slopeAll;           % Column 10: CPP Slope
participant_level(:,11) = CPP_Amplitude;          % Column 11: CPP Amplitude
participant_level(:,12) = FrontalNeg_slope';      % Column 12: Frontal Negativity Slope (CNV)
participant_level(:,13) = frontalNeg_Amp;         % Column 13: CNV Amplitude

% Create the table starting from column 1 of participant_level
T = table(subject_folder', ...
    participant_level(:,1), participant_level(:,2), participant_level(:,3), participant_level(:,4), ...
    participant_level(:,5), participant_level(:,6), participant_level(:,7), participant_level(:,8), ...
    participant_level(:,9), participant_level(:,10), participant_level(:,11), participant_level(:,12), ...
    participant_level(:,13), ...
    'VariableNames', {'IDs', 'Group', 'site', 'RT', 'hitrate', 'misses', ...
    'PreAlpha_collapsed', 'N2CLatency', 'N2c_indiv', 'originalCppOnset', 'CPP_slopeAll', 'CPP_Amplitude',...
    'FrontalNeg_slope', 'frontalNeg_Amp'});

writetable(T,[pathOut,'GitHub_participant_level_matrix.csv'])