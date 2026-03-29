function result = run_repro_asset(asset_name, out_json)
%RUN_REPRO_ASSET Execute a repo asset under the clean reproduction path.

if nargin < 2
    out_json = '';
end

info = setup_repro_paths();
result = struct();
result.asset = asset_name;
result.timestamp = char(datetime('now', 'TimeZone', 'UTC', 'Format', 'yyyy-MM-dd''T''HH:mm:ss''Z'''));
result.status = 'PASS';
result.fixed = false;
result.environment = info;

try
    switch asset_name
        case 'add_paths'
            paths_added = add_paths();
            payload = struct('num_paths', numel(paths_added));
        case 'ar2exp'
            payload = summarize_value(ar2exp([0.95, 0.02]));
        case 'bin_data'
            data = sparse([1, 2, 3, 4; 5, 6, 7, 8]);
            payload = summarize_value(bin_data(data, 1, 2));
        case 'create_heart_mask'
            payload = summarize_value(create_heart_mask([64, 96]));
        case 'get_random_points'
            [y, x] = find(create_heart_mask([64, 96]));
            [X, Y] = get_random_points(x, y, 5, 10, 0);
            payload = struct('num_points', numel(X), 'first_point', [X(1), Y(1)]);
        case 'v2uint8'
            payload = summarize_value(v2uint8(randn(8, 8, 3), [-1, 1]));
        case 'sim_drifting_activities'
            payload = summarize_value(sim_drifting_activities(12, 4));
        case 'create_random_spike'
            P = 0.2 * ones(5, 1);
            [S, LFP] = create_random_spike(5, 20, 10, P, 0, 1, 1);
            payload = struct('S', summarize_value(S), 'LFP', summarize_value(LFP));
        case 'create_neuron_data'
            opt = default_opt();
            [Aout, bA, C, S, LFP] = create_neuron_data(opt);
            payload = struct( ...
                'Aout', summarize_value(Aout), ...
                'bA', summarize_value(bA), ...
                'C', summarize_value(C), ...
                'S', summarize_value(S), ...
                'LFP', summarize_value(LFP));
        case 'Add_NRmotion'
            [dM, dA, dbA, D] = Add_NRmotion(rand(80, 90), rand(80, 90, 3), rand(80, 90, 3), 2, 0, 1, 20);
            payload = struct( ...
                'dM', summarize_value(dM), ...
                'dA', summarize_value(dA), ...
                'dbA', summarize_value(dbA), ...
                'D', summarize_value(D));
        case 'save_as_avi'
            avi_path = fullfile(info.runtime_dir, 'asset_save_as_avi.avi');
            if exist(avi_path, 'file')
                delete(avi_path);
            end
            save_as_avi(uint8(randi(255, [16, 16, 4])), avi_path);
            payload = struct('path', avi_path, 'exists', isfile(avi_path));
        case 'saveash5'
            h5_path = fullfile(info.runtime_dir, 'asset_saveash5.h5');
            if exist(h5_path, 'file')
                delete(h5_path);
            end
            saveash5(uint8(randi(255, [16, 16, 4])), h5_path);
            payload = struct('path', h5_path, 'exists', isfile(h5_path));
        case 'BL_sim'
            mr = load(resolve_repo_data_file('BL.mat', info.repo_root), 'Mr');
            payload = summarize_value(BL_sim(mr.Mr));
        case 'estimate_time_constant_PV'
            payload = summarize_value(estimate_time_constant_PV(rand(8, 50)));
        case 'Simulate_Ca_video'
            [pth_out, outpath, Mot, file_name, A, GT_motion_xy] = Simulate_Ca_video( ...
                'Nneu', 10, 'ses', 1, 'F', 20, 'save_videos', false, ...
                'save_mat', false, 'save_GT', false, 'plotme', 0, ...
                'session_motion_std', 0);
            payload = struct( ...
                'pth_out', summarize_value(pth_out), ...
                'outpath', summarize_value(outpath), ...
                'Mot', summarize_value(Mot), ...
                'file_name', summarize_value(file_name), ...
                'A', summarize_value(A), ...
                'GT_motion_xy', summarize_value(GT_motion_xy));
        case 'README_example_1'
            [pth_out, outpath, Mot, file_name, A, GT_motion_xy] = Simulate_Ca_video( ...
                'Nneu', 30, 'ses', 2, 'CA1_A', true, 'PNR', 1, 'F', 20, ...
                'save_videos', false, 'save_mat', false, 'save_GT', false, ...
                'plotme', 0);
            payload = struct( ...
                'pth_out', summarize_value(pth_out), ...
                'outpath', summarize_value(outpath), ...
                'Mot', summarize_value(Mot), ...
                'file_name', summarize_value(file_name), ...
                'A', summarize_value(A), ...
                'GT_motion_xy', summarize_value(GT_motion_xy));
        case 'README_example_2'
            [pth_out, outpath, Mot, file_name, A, GT_motion_xy] = Simulate_Ca_video( ...
                'Nneu', 20, 'ses', 1, 'F', 60, 'LFP', 8, ...
                'spike_prob', [-4.91, 0.83], 'sf', 60, ...
                'save_videos', false, 'save_mat', false, 'save_GT', false, ...
                'plotme', 0);
            payload = struct( ...
                'pth_out', summarize_value(pth_out), ...
                'outpath', summarize_value(outpath), ...
                'Mot', summarize_value(Mot), ...
                'file_name', summarize_value(file_name), ...
                'A', summarize_value(A), ...
                'GT_motion_xy', summarize_value(GT_motion_xy));
        case 'README_example_3'
            [pth_out, outpath, Mot, file_name, A, GT_motion_xy] = Simulate_Ca_video( ...
                'Nneu', 40, 'ses', 4, 'F', 40, 'NR_misalignment', 0, ...
                'min_dist', 2, 'spike_prob', [-2, 0.83], 'A2', true, ...
                'overlap', 0.2, 'save_videos', false, 'save_mat', false, ...
                'save_GT', false, 'plotme', 0);
            payload = struct( ...
                'pth_out', summarize_value(pth_out), ...
                'outpath', summarize_value(outpath), ...
                'Mot', summarize_value(Mot), ...
                'file_name', summarize_value(file_name), ...
                'A', summarize_value(A), ...
                'GT_motion_xy', summarize_value(GT_motion_xy));
        otherwise
            error('Unsupported asset "%s".', asset_name);
    end

    result.payload = payload;
catch ME
    result.status = 'FAIL';
    result.error = getReport(ME, 'extended', 'hyperlinks', 'off');
end

if ~isempty(out_json)
    out_dir = fileparts(out_json);
    if ~isempty(out_dir) && ~isfolder(out_dir)
        mkdir(out_dir);
    end
    fid = fopen(out_json, 'w');
    if fid == -1
        error('Failed to open "%s" for writing.', out_json);
    end
    cleaner = onCleanup(@() fclose(fid));
    fprintf(fid, '%s\n', jsonencode(result, 'PrettyPrint', true));
end
end

function out = summarize_value(value)
if isnumeric(value) || islogical(value)
    out = struct('class', class(value), 'size', size(value), 'is_complex', ~isreal(value));
    if isempty(value)
        out.is_empty = true;
    elseif isreal(value)
        out.min = min(double(value(:)));
        out.max = max(double(value(:)));
    else
        out.min_abs = min(abs(double(value(:))));
        out.max_abs = max(abs(double(value(:))));
        out.mean_real = mean(real(double(value(:))));
        out.mean_imag = mean(imag(double(value(:))));
    end
elseif ischar(value) || (isstring(value) && isscalar(value))
    out = char(string(value));
elseif iscell(value)
    out = struct('class', class(value), 'size', size(value), 'numel', numel(value));
elseif isstruct(value)
    out = struct('class', class(value), 'fields', {fieldnames(value)});
else
    out = struct('class', class(value), 'size', size(value));
end
end

function opt = default_opt()
opt = struct();
opt.create_mask = true;
opt.d = [80, 100];
opt.min_dist = 8;
opt.Nneu = 8;
opt.plotme = 0;
opt.CA1_A = false;
opt.A2 = false;
opt.ses = 2;
opt.scale = 1;
opt.spike_prob = [-4.9, 2.25];
opt.sf = 10;
opt.invtRise = [2.08, 0.29];
opt.invtDecay = [0.55, 0.44];
opt.F = 20;
opt.LFP = 0;
opt.force_active = 1;
end
