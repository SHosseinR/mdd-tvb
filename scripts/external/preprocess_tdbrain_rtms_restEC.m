% Copy of preprocess_tdbrain_mdd_healthy_restEC.m (graph-opt project) restricted to the
% TDBRAIN MDD-rTMS cohort. Only the dataset root, output folder and group selection differ.
%% ================== SETTINGS ==================
clc;
clearvars -except max_per_group overwrite_existing dry_run;

dataset_root = 'D:/university/projects/graph-opt/tbdbrain/TDBRAIN_Dataset_V3_1';

participants_file = fullfile(dataset_root, 'TDBRAIN_participants_V3.xlsx');
root_out = fullfile(dataset_root, 'output', 'preprocessed_tdbrain_restEC_rtms');

groups = {'rTMS'};
task_name = 'restEC';   % eye closed
session_id = 1;

chanNames = { ...
    'Fp1','Fp2','F7','F3','Fz','F4','F8', ...
    'FC3','FCz','FC4','T7','C3','Cz','C4','T8', ...
    'CP3','CPz','CP4','P7','P3','Pz','P4','P8', ...
    'O1','Oz','O2'};

margin_sec = 4;          % expand removal region +/- margin_sec
amp_thresh = 10000;      % uV, matching the old ADHD script
min_retained_sec = 60;   % subject-level QC: keep at least half of 120 s rest
max_reject_fraction = 0.50;

if ~exist('max_per_group', 'var') || isempty(max_per_group)
    max_per_group = inf;
end
if ~exist('overwrite_existing', 'var') || isempty(overwrite_existing)
    overwrite_existing = false;
end
if ~exist('dry_run', 'var') || isempty(dry_run)
    dry_run = false;
end

if ~exist(root_out, 'dir')
    mkdir(root_out);
end

%% ================== EEGLAB ==================
if isempty(which('eeglab'))
    eeglab_path = 'D:\matlab_modules\eeglab';
    if exist(eeglab_path, 'dir')
        addpath(eeglab_path);
    end
end

[ALLEEG, EEG, CURRENTSET, ALLCOM] = eeglab('nogui'); %#ok<ASGLU>

%% ================== LOAD PARTICIPANTS ==================
opts = detectImportOptions(participants_file, 'VariableNamingRule', 'preserve');
T = readtable(participants_file, opts);

ids = clean_column(T, 'TDBRAIN_ID');
indication = clean_column(T, 'indication');
formal_status = clean_column(T, 'formal_status');
dataset_name = clean_column(T, 'Dataset');
sess = str2double(clean_column(T, 'sessID'));

groupRows = struct();

% Plain, non-rTMS MDD. formal_status is UNKNOWN for many non-rTMS MDD rows,
% so indication is the usable diagnosis field here.
groupRows.MDD = strcmpi(indication, 'MDD') & ...
    ~strcmpi(dataset_name, 'MDD-rTMS') & sess == session_id;

% External test cohort (added 2026-09-25): rTMS-treated MDD patients,
% baseline (session 1) recordings.  Identical preprocessing to MDD/Healthy.
groupRows.rTMS = strcmpi(dataset_name, 'MDD-rTMS') & sess == session_id;

% Clean controls.
groupRows.Healthy = strcmpi(indication, 'HEALTHY') & ...
    strcmpi(formal_status, 'HEALTHY') & sess == session_id;

all_rejects = {};
summary_rows = {};

fprintf('\nTD-BRAIN %s preprocessing\n', task_name);
fprintf('Output root: %s\n', root_out);
fprintf('max_per_group: %g | overwrite_existing: %d | dry_run: %d\n', ...
    max_per_group, overwrite_existing, dry_run);

%% ================== LOOP ==================
for g = 1:numel(groups)
    group = groups{g};
    rows = groupRows.(group);
    subjectIDs = unique(ids(rows), 'stable');
    subjectIDs(subjectIDs == "") = [];

    if isfinite(max_per_group)
        subjectIDs = subjectIDs(1:min(numel(subjectIDs), max_per_group));
    end

    outFolder = fullfile(root_out, group);
    if ~exist(outFolder, 'dir')
        mkdir(outFolder);
    end

    fprintf('\n===== GROUP: %s | selected rows/subjects: %d/%d =====\n', ...
        group, nnz(rows), numel(subjectIDs));

    for f = 1:numel(subjectIDs)
        subjectID = char(subjectIDs(f));
        name = subjectID;

        eeg_file = fullfile(dataset_root, subjectID, sprintf('ses-%d', session_id), 'eeg', ...
            sprintf('%s_ses-%d_task-%s_eeg.bdf', subjectID, session_id, task_name));
        saveFolder = fullfile(outFolder, name);
        set_file = fullfile(saveFolder, [name '.set']);

        fprintf('\n[%s %d/%d] Processing: %s\n', group, f, numel(subjectIDs), subjectID);

        if exist(set_file, 'file') && ~overwrite_existing
            fprintf('Already exists, skipping: %s\n', set_file);
            summary_rows(end+1, :) = {group, subjectID, eeg_file, set_file, ...
                "skipped_existing", 0, 0, 0, 0, 0, 0, ""}; %#ok<SAGROW>
            continue;
        end

        if ~exist(eeg_file, 'file')
            fprintf('Missing BDF: %s\n', eeg_file);
            summary_rows(end+1, :) = {group, subjectID, eeg_file, "", ...
                "missing_bdf", 0, 0, 0, 0, 0, 0, "BDF file not found"}; %#ok<SAGROW>
            continue;
        end

        if dry_run
            summary_rows(end+1, :) = {group, subjectID, eeg_file, set_file, ...
                "dry_run_selected", 0, 0, 0, 0, 0, 0, ""}; %#ok<SAGROW>
            continue;
        end

        try
            %% ---------- Load ----------
            EEG = pop_biosig(eeg_file);
            EEG.setname = name;
            EEG = eeg_checkset(EEG);

            for c = 1:numel(EEG.chanlocs)
                EEG.chanlocs(c).labels = strtrim(EEG.chanlocs(c).labels);
            end

            chLabels = string({EEG.chanlocs.labels});
            missingCh = setdiff(string(chanNames), chLabels);
            if ~isempty(missingCh)
                reason = "Missing EEG channels: " + strjoin(missingCh, ", ");
                fprintf('%s\n', reason);
                summary_rows(end+1, :) = {group, subjectID, eeg_file, "", ...
                    "skipped_missing_channels", 0, 0, 0, 0, 0, 0, reason}; %#ok<SAGROW>
                continue;
            end

            EEG = pop_select(EEG, 'channel', chanNames);
            EEG = eeg_checkset(EEG);

            original_pnts = EEG.pnts;
            original_sec = EEG.pnts / EEG.srate;

            %% ---------- 1) 50 Hz NOTCH ----------
            EEG = pop_eegfiltnew(EEG, 'locutoff', 49, 'hicutoff', 51, 'revfilt', 1);

            %% ---------- 2) BANDPASS ----------
            EEG = pop_eegfiltnew(EEG, 'locutoff', 1);
            EEG = pop_eegfiltnew(EEG, 'hicutoff', 60);
            EEG = eeg_checkset(EEG);

            %% ---------- 3) MARK + REMOVE HIGH AMPLITUDE SEGMENTS ----------
            fs = EEG.srate;
            margin = round(margin_sec * fs);

            data = double(EEG.data);
            badMask = any(abs(data) > amp_thresh, 1) | any(~isfinite(data), 1);

            badIdx = find(badMask);
            removeMask = false(1, size(data, 2));

            for i = 1:numel(badIdx)
                s1 = max(1, badIdx(i) - margin);
                e1 = min(size(data, 2), badIdx(i) + margin);
                removeMask(s1:e1) = true;
            end

            d = diff([0 removeMask 0]);
            segStarts = find(d == 1);
            segEnds = find(d == -1) - 1;

            if ~isempty(segStarts)
                for k = 1:numel(segStarts)
                    EEG.event(end+1).type = 'amp_reject';
                    EEG.event(end).latency = segStarts(k);
                    EEG.event(end).duration = segEnds(k) - segStarts(k) + 1;
                    EEG.event(end).urevent = [];
                end
                EEG = eeg_checkset(EEG, 'eventconsistency');
                EEG = eeg_checkset(EEG, 'makeur');
            end

            rej = struct();
            rej.start_samp = segStarts(:);
            rej.end_samp = segEnds(:);
            rej.dur_samp = (segEnds(:) - segStarts(:) + 1);
            rej.start_sec = (rej.start_samp - 1) / fs;
            rej.end_sec = (rej.end_samp - 1) / fs;
            rej.dur_sec = rej.dur_samp / fs;

            keepMask = ~removeMask;
            retained_pnts = nnz(keepMask);
            retained_sec = retained_pnts / fs;
            rejected_sec = nnz(removeMask) / fs;
            reject_fraction = nnz(removeMask) / numel(removeMask);

            entry = struct();
            entry.group = group;
            entry.name = name;
            entry.file_bdf = eeg_file;
            entry.rejects = rej;
            entry.original_sec = original_sec;
            entry.retained_sec = retained_sec;
            entry.reject_fraction = reject_fraction;

            if retained_sec < min_retained_sec || reject_fraction > max_reject_fraction
                entry.status = 'skipped_qc';
                all_rejects{end+1} = entry; %#ok<SAGROW>
                reason = sprintf('retained %.2f sec, rejected %.1f%%', retained_sec, 100 * reject_fraction);
                fprintf('QC skip: %s\n', reason);
                summary_rows(end+1, :) = {group, subjectID, eeg_file, "", ...
                    "skipped_qc", original_sec, retained_sec, rejected_sec, ...
                    reject_fraction, numel(segStarts), EEG.nbchan, string(reason)}; %#ok<SAGROW>
                continue;
            end

            EEG.data = EEG.data(:, keepMask);
            EEG.pnts = size(EEG.data, 2);
            EEG.xmin = 0;
            EEG.xmax = (EEG.pnts - 1) / fs;
            EEG.times = (0:EEG.pnts-1) / fs * 1000;
            EEG = eeg_checkset(EEG, 'eventconsistency');

            %% ---------- 4) AVERAGE RE-REFERENCE ----------
            EEG = pop_reref(EEG, []);
            EEG = eeg_checkset(EEG);

            %% ---------- Save .set ----------
            if ~exist(saveFolder, 'dir')
                mkdir(saveFolder);
            end

            EEG = pop_saveset(EEG, 'filename', [name '.set'], 'filepath', saveFolder);

            entry.status = 'saved';
            entry.file_set = set_file;
            all_rejects{end+1} = entry; %#ok<SAGROW>

            fprintf('Saved -> %s | retained %.2f/%.2f sec | rejects: %d\n', ...
                saveFolder, retained_sec, original_sec, numel(segStarts));

            summary_rows(end+1, :) = {group, subjectID, eeg_file, set_file, ...
                "saved", original_sec, retained_sec, rejected_sec, ...
                reject_fraction, numel(segStarts), EEG.nbchan, ""}; %#ok<SAGROW>

        catch ME
            fprintf(2, 'FAILED: %s\n', ME.message);
            summary_rows(end+1, :) = {group, subjectID, eeg_file, "", ...
                "failed", 0, 0, 0, 0, 0, 0, string(ME.message)}; %#ok<SAGROW>
        end
    end
end

%% ================== SAVE GLOBAL LOGS ==================
if ~exist(root_out, 'dir')
    mkdir(root_out);
end

save(fullfile(root_out, 'all_rejects.mat'), 'all_rejects');

summary = cell2table(summary_rows, 'VariableNames', { ...
    'group', 'subject_id', 'file_bdf', 'file_set', 'status', ...
    'original_sec', 'retained_sec', 'rejected_sec', 'reject_fraction', ...
    'n_reject_segments', 'n_channels', 'note'});
writetable(summary, fullfile(root_out, 'processing_summary.csv'));

disp('===== FINISHED =====');
disp(['Saved global reject file -> ' fullfile(root_out, 'all_rejects.mat')]);
disp(['Saved processing summary -> ' fullfile(root_out, 'processing_summary.csv')]);

%% ================== HELPERS ==================
function out = clean_column(T, varName)
    vals = T.(varName);
    if iscell(vals)
        out = strings(size(vals));
        for i = 1:numel(vals)
            out(i) = clean_one(vals{i});
        end
    else
        out = strings(size(vals));
        for i = 1:numel(vals)
            out(i) = clean_one(vals(i));
        end
    end
end

function s = clean_one(v)
    if iscell(v)
        v = v{1};
    end
    if ismissing(v)
        s = "";
        return;
    end
    if isnumeric(v)
        if isnan(v)
            s = "";
        else
            s = string(v);
        end
    else
        s = string(v);
    end
    s = strtrim(s);
    if strcmpi(s, "nan") || strcmpi(s, "<missing>")
        s = "";
    end
end
