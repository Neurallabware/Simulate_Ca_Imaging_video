# MATLAB Toolbox and Compatibility Workarounds

- `rng` / `rand` / `randn` / `randperm`: wrapped with MATLAB-order helpers in `simulate_ca_imaging_video._compat` so multi-dimensional draws follow MATLAB's column-major semantics.
- `imresize`: replaced with `skimage.transform.resize`; small residual differences remain because MATLAB's bicubic kernel is not bit-identical.
- `imwarp` / `imtranslate`: replaced with `skimage.transform.warp` and `scipy.ndimage.map_coordinates` while preserving range and disabling clipping to match MATLAB behavior.
- `VideoWriter('Uncompressed AVI')`: replaced with `imageio` plus `imageio-ffmpeg` using `ffv1` to produce a readable AVI in Python-only environments.
- `envelope`: approximated with the Hilbert analytic signal in `create_sine`; spike/LFP traces match, while the derived envelope is treated as a bounded behavioral equivalent.
- `addpath(genpath(...))`: replaced with `add_paths()` recursion that omits `.git` and keeps runtime path setup explicit.
- Vendored MATLAB `ProgressBar`: intentionally not mirrored 1:1; Python runtime uses a thin `tqdm` wrapper in `progress.py`.
- MATLAB Engine is used only for parity testing and report generation. Package runtime and notebook execution do not require MATLAB.
