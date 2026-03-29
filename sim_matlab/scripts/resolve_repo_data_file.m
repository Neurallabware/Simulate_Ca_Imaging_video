function path_out = resolve_repo_data_file(filename, repo_root)
%RESOLVE_REPO_DATA_FILE Locate a repo data file under data/ with root fallback.

if nargin < 2 || isempty(repo_root)
    script_dir = fileparts(mfilename('fullpath'));
    repo_root = fileparts(fileparts(script_dir));
end

candidate = fullfile(repo_root, 'data', filename);
if isfile(candidate)
    path_out = candidate;
    return;
end

fallback = fullfile(repo_root, filename);
if isfile(fallback)
    path_out = fallback;
    return;
end

path_out = candidate;
end
