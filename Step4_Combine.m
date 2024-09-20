% Clear workspace and close all figures
clear;
close all;
clc;

%% Settings
% --------------------------------------------------------------------------------------------------
% Paths to code and data
% --------------------------------------------------------------------------------------------------
scriptPath = '/Users/manab/GoogleDrive_Gmail/Code/GitHub_ADHD_DM';
addpath(genpath('/Users/manab/Desktop/Functions/eeglab2021.0/'));
eeglab;
pathOut = '/Users/manab/Desktop/ADHDKids/Analysis2023/dataScriptsResults/Data_temp/';
path_ADHD = pathOut;
path_Control = pathOut;
cd(pathOut);

% Name of subject folders (Control and ADHD)
subject_folderControl = {'C133', 'C134', 'C135', 'C136', 'C137', 'C138', 'C139', 'C141', 'C500', 'C501', 'C502', ...
    'C503', 'C504', 'C505', 'C506', 'C507', 'C508', 'C509', 'C510', 'C511', 'C512', '001_KK', 'C12', 'C21', ...
    '002_LK', 'C50', 'C131', 'C20', 'C132', 'C213', 'C204', 'C171', 'C13', 'C119', 'C121', 'C194', 'C168', ...
    'C14', 'C49', 'C400', 'C100', 'C117', 'C115', 'C110', 'C183', 'C144', 'C252', 'C140', 'C259', 'C143', ...
    'C89', 'C62', 'C88', 'C84', 'C87'};

subject_folderADHD = {'AD88C', 'AD91C', 'AD92C', 'AD93C', 'AD94C', 'AD96C', 'AD33C', 'AD34C', 'AD35C', 'AD36C', 'AD37C', ...
    'AD81C', 'AD82C', 'AD83C', 'AD85C', 'AD84C', 'AD86C', 'AD40C', 'AD11C', 'AD43C', 'AD18C', 'AD24C', 'AD52C', ...
    'AD5C', 'AD54C', 'AD75C', 'AD15C', 'AD49C', 'AD69C', 'AD32C', 'AD16C', 'AD72C', 'AD59C', 'AD27C', 'AD46C', ...
    'AD22C', 'AD26C', 'AD99C', 'AD48C', 'AD57C', 'AD51C', 'AD19C', 'AD98C', 'AD25C', 'AD23C'};

% Combine ADHD and Control subject folders
subject_folder = [subject_folderADHD, subject_folderControl];

% --------------------------------------------------------------------------------------------------
% Monash IDs (Control and ADHD)
monashIds = {'AD88C', 'AD91C', 'AD92C', 'AD93C', 'AD94C', 'AD96C', 'AD33C', 'AD34C', 'AD35C', ...
    'AD36C', 'AD37C', 'AD81C', 'AD82C', 'AD83C', 'AD85C', 'AD84C', 'AD86C', 'C133', 'C134', 'C135', ...
    'C136', 'C137', 'C138', 'C139', 'C141', 'C500', 'C501', 'C502', 'C503', 'C504', 'C505', 'C506', ...
    'C507', 'C508', 'C509', 'C510', 'C511', 'C512'};

%% Exclude subjects
exclSubj = ismember(subject_folder, {'AD34C', 'C503', 'C510', 'AD94C', 'AD33C', 'C133', 'C134', 'C505', 'C506', ...
    'C511', 'AD93C', 'AD96C', 'C504', 'C512', 'AD37C'});
subject_folder(exclSubj) = [];

%% Data Settings
fs = 500; % Sampling frequency
numch = 64; % Number of channels
rtlim = [0.2, 1.88]; % RT limits in seconds (200ms to 1880ms)
[B, A] = butter(4, 8*2/fs); % 8 Hz butterworth filter

% Target codes
targcodes = [101, 103, 105; 102, 104, 106]; % left patch, right patch

% Time settings for ERP epoch
ts = -0.8*fs:1.880*fs;
t = ts * 1000/fs; % Time in ms

% Baseline interval in ms
BLint = [-100, 0];

%% Initialize variables
s = 0;  % Subject index counter
kk = 0;  % DDM index counter
kkF = 0; % False alarm index counter

% Loop through all subjects
for singleParticipant = 1:length(subject_folder)
    
    s = s + 1; % Increment subject counter
    
    % Determine group (ADHD or Control) and load corresponding data
    if contains(subject_folder{singleParticipant}, 'AD')
        group = 1;  % ADHD group
        groupName = 'ADHD';
        group_vector(s) = group;
        dataFile = sprintf('Data_temp/DM_beh_%s_ADHD.mat', subject_folder{singleParticipant});
    else
        group = 2;  % Control group
        groupName = 'Control';
        group_vector(s) = group;
        dataFile = sprintf('Data_temp/DM_beh_%s_Control.mat', subject_folder{singleParticipant});
    end
    
    load(dataFile, 'trialCount', 'trialCount_left', 'trialCount_Right', 'ERP_cond', 'ERPr_cond', ...
        'N2_cond', 'Alpha_cond_pre', 'Alpha_cond_post', 'Pupil_cond_pre', 'Pupil_cond_post', ...
        'CPP_patch_onsets', 'avRT', 'RT_Zs', 'allTrueResponseLeft', 'allTrueResponseRight', ...
        'RT_index', 'RT', 'RT_Left', 'RT_Right', 'RT_Assymetry', 'ValidPreAlphaTrials', ...
        'ValidN2Trials', 'ValidCPPTrials', 'Valid_RT_Trials', 'UQMon','RT_DMM', 'RT_Z_DDM', 'ACC_DDM', ...
        'Alpha_smooth_time', 'false_alarm');

    %% Store data for each participant
    % Trial counts
    trialCountAll(s) = trialCount;
    trialCount_leftAll(s) = trialCount_left;
    trialCount_RightAll(s) = trialCount_Right;
    
    % Group and UQMon variables
    groupAll(s) = group;
    UQMonAll(s) = UQMon;

    % Store ERP, N2, Alpha, Pupil, and CPP data
    ERP_condAll(s, :, :, :, :) = squeeze(ERP_cond(1, :, :, :, :, group));
    ERPr_condAll(s, :, :, :, :) = squeeze(ERPr_cond(1, :, :, :, :, group));
    N2_condAll(s, :, :, :, :) = squeeze(N2_cond(1, :, :, :, :, group));
    Alpha_cond_preAll(s, :, :, :, :) = squeeze(Alpha_cond_pre(1, :, :, :, :, group));
    Alpha_cond_postAll(s, :, :, :, :) = squeeze(Alpha_cond_post(1, :, :, :, :, group));
    Pupil_cond_preAll(s, :, :, :) = squeeze(Pupil_cond_pre(1, :, :, :, group));
    Pupil_cond_postAll(s, :, :, :) = squeeze(Pupil_cond_post(1, :, :, :, group));
    CPP_patch_onsetsAll(s, :) = squeeze(CPP_patch_onsets(1, :));
    
    % Store RT, false alarms, and DDM data for each patch and condition
    for patch = 1:2
        for iti = 1:3
            avRTAll{s, patch, iti} = avRT{1, patch, iti};
            RT_zsAll{s, patch, iti} = RT_Zs{1, patch, iti};
            false_alarmAll{s, patch, iti} = false_alarm{1, patch, iti};
        end
    end
    
    % DDM data
    for patch = 1:2
        for iti = 1:3
            for i = 1:length(RT_DMM{1, patch, iti})
                kk = kk + 1;
                DDM.subj_idx(kk) = s;
                DDM.side(kk) = patch;
                DDM.iti(kk) = iti;
                DDM.group(kk) = group;
                DDM.rt(kk) = RT_DMM{1, patch, iti}(i);
                DDM.response(kk) = ACC_DDM{1, patch, iti}(i) / 2;
                DDM.RTZ(kk) = RT_Z_DDM{1, patch, iti}(i);
            end
        end
    end
    
    % False alarm data
    for patch = 1:2
        for iti = 1:3
            for i = 1:length(false_alarm{1, patch, iti})
                kkF = kkF + 1;
                FSAlarm.subj_idx(kkF) = s;
                FSAlarm.side(kkF) = patch;
                FSAlarm.iti(kkF) = iti;
                FSAlarm.group(kkF) = group;
                FSAlarm.falseAlarm(kkF) = false_alarm{1, patch, iti}(i);
            end
        end
    end
    
    % Store response data
    allTrueResponseLeftAll(s) = allTrueResponseLeft;
    allTrueResponseRightAll(s) = allTrueResponseRight;
    RT_indexAll(s) = RT_index;
    RTAll(s) = RT;
    RT_LeftAll(s) = RT_Left;
    RT_RightAll(s) = RT_Right;
    RT_AssymetryAll(s) = RT_Assymetry;
    ValidPreAlphaTrialsAll(s) = ValidPreAlphaTrials;
    ValidN2TrialsAll(s) = ValidN2Trials;
    ValidCPPTrialsAll(s) = ValidCPPTrials;
    Valid_RT_TrialsAll(s) = Valid_RT_Trials;
    
end

%% Convert struct fields to columns for saving
fields = fieldnames(DDM);
for i = 1:length(fields)
    DDM.(fields{i}) = DDM.(fields{i})';
end

fieldsFS = fieldnames(FSAlarm);
for i = 1:length(fieldsFS)
    FSAlarm.(fieldsFS{i}) = FSAlarm.(fieldsFS{i})';
end

%% Save the results
save([pathOut, 'DM_beh.mat'], 'subject_folderControl', 'subject_folderADHD', 'monashIds', ...
    'subject_folder', 'trialCountAll', 'trialCount_leftAll', 'trialCount_RightAll', 'ERP_condAll', ...
    'ERPr_condAll', 'N2_condAll', 'Alpha_cond_preAll', 'Alpha_cond_postAll', 'Pupil_cond_preAll', ...
    'Pupil_cond_postAll', 'CPP_patch_onsetsAll', 'avRTAll', 'RT_zsAll', 'allTrueResponseLeftAll', ...
    'allTrueResponseRightAll', 'RT_indexAll', 'RTAll', 'RT_LeftAll', 'RT_RightAll', 'RT_AssymetryAll', ...
    'ValidPreAlphaTrialsAll', 'ValidN2TrialsAll', 'ValidCPPTrialsAll', 'Valid_RT_TrialsAll', 'groupAll', 'Alpha_smooth_time');

% Save CSV data
T = table(subject_folder', groupAll', CPP_patch_onsetsAll(:, 1), CPP_patch_onsetsAll(:, 2), ...
    'VariableNames', {'IDs', 'Group', 'CPPonsetLeft', 'CPPonsetRight'});
writetable(T, [pathOut, 'CPPonset.csv']);

save([pathOut, 'falsealarm.mat'], 'false_alarmAll');

% Export data to CSV files
struct2csv(FSAlarm', [pathOut, 'dataFalseAlarm.csv']);
struct2csv(DDM', [pathOut, 'dataBehaviour.csv']);



