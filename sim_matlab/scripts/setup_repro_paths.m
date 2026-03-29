function info = setup_repro_paths(repo_root)
%SETUP_REPRO_PATHS Configure a clean MATLAB path for reproducible runs.

if nargin < 1 || isempty(repo_root)
    repo_root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
end

restoredefaultpath;

matlab_root = fullfile(repo_root, 'sim_matlab');
data_root = fullfile(repo_root, 'data');
repo_paths = split_paths(genpath(matlab_root));
repo_paths = repo_paths(~contains(repo_paths, [filesep '.git']));
repo_paths = repo_paths(~contains(repo_paths, [filesep '.venv']));

external_paths = {
    '/home/yz/ca_pipeline/CaliAli/Other_codes'
    '/home/yz/ca_pipeline/CaliAli/Other_codes/Vesselness_Filter'
    '/home/yz/ca_pipeline/CaliAli/CNMF-e/OASIS_matlab/functions'
};

existing_external = external_paths(cellfun(@isfolder, external_paths));
missing_external = external_paths(~cellfun(@isfolder, external_paths));

optional_paths = {};
if isfolder(data_root)
    optional_paths{end + 1, 1} = data_root; %#ok<AGROW>
end
paths_added = unique([repo_paths; optional_paths; existing_external], 'stable');
addpath(paths_added{:});

data_files = {'A.mat', 'A2.mat', 'A_CA1.mat', 'BL.mat'};
data_status = struct();
for k = 1:numel(data_files)
    data_status.(matlab.lang.makeValidName(data_files{k})) = ...
        resolve_repo_data_file(data_files{k}, repo_root);
end

function_status = struct( ...
    'Simulate_Ca_video', which('Simulate_Ca_video'), ...
    'BL_sim', which('BL_sim'), ...
    'estimate_time_constant_PV', which('estimate_time_constant_PV'), ...
    'vesselness_PV', which('vesselness_PV'), ...
    'estimate_time_constant', which('estimate_time_constant'));

runtime_dir = fullfile(repo_root, 'artifacts', 'runtime');
if ~isfolder(runtime_dir)
    mkdir(runtime_dir);
end

info = struct();
info.repo_root = repo_root;
info.matlab_root = matlab_root;
info.data_root = data_root;
info.paths_added = paths_added;
info.external_paths = existing_external;
info.missing_external_paths = missing_external;
info.data_files = data_status;
info.function_status = function_status;
info.runtime_dir = runtime_dir;
end

function paths = split_paths(raw_paths)
parts = strsplit(raw_paths, pathsep);
parts = parts(~cellfun(@isempty, parts));
paths = parts(:);
end
