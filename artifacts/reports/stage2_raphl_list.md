# Final RAPHL List

## Stage 1: MATLAB Reproduction

- `[✅ DONE] S1.1` Read repo config/setup surfaces: `README.md`, repo layout, helper scripts, vendored `ProgressBar`
- `[✅ DONE] S1.2` Identify MATLAB version, toolbox dependencies, and hardware requirements
- `[✅ DONE] S1.3` Catalog data dependencies and file formats
- `[✅ DONE] S1.4` Catalog executable assets and external MATLAB gaps
- `[✅ DONE] S1.5` Reconnaissance Report gate passed
- `[✅ DONE] S1.6` Confirm no repo-declared external downloads are required
- `[✅ DONE] S1.7` Record repo `.mat` files as the primary data source under the repo-as-source policy
- `[⚠️ FIXED] S1.8` Create a clean MATLAB launcher/path patch using `restoredefaultpath`, repo paths, and exact CaliAli dependency paths
- `[✅ DONE] S1.9` Mark login-gated assets as `MANUAL_DOWNLOAD_REQUIRED` if discovered; none were required
- `[✅ DONE] S1.10` Gate: verify repo data paths and CaliAli dependency paths resolve correctly
- `[✅ DONE] S1.11` Run test suites and capture pass/fail counts; recorded `N/A` because no MATLAB test suite exists
- `[✅ DONE] S1.12` Run executable examples: README examples, `Simulate_Ca_video`, `BL_sim`, and `estimate_time_constant_PV`
- `[✅ DONE] S1.13` Run notebooks/live scripts and capture pass/fail; recorded `N/A` because none exist
- `[⚠️ FIXED] S1.14` Diagnose, fix, rerun, and document Stage 1 failures via wrapper/path patch
- `[✅ DONE] S1.15` Gate: every executable asset has `PASS`, `FIXED`, or `UNRESOLVED`
- `[✅ DONE] S1.16` Write Data Manifest with file, size, source, and local path
- `[✅ DONE] S1.17` Write Stage 1 Verdict
- `[✅ DONE] S1.18` Reprint full Stage 1 RAPHL list with final statuses

## Stage 2: MATLAB to Python

### Planning

- `[✅ DONE] S2.1` Map all MATLAB files, dependencies, and call graph
- `[✅ DONE] S2.2` Identify shared utilities, data I/O, visualization, and external CaliAli dependencies
- `[✅ DONE] S2.3` Propose Python project structure under `src/simulate_ca_imaging_video/` with CaliAli mirrors under `cali_ali_mirror/`
- `[✅ DONE] S2.4` Generate conversion order and effort estimates
- `[✅ DONE] S2.5` Build dedicated `uv` environment and install runtime dependencies plus MATLAB Engine

### Conversion Queue

- `[⚠️ FIXED] S2.C-01` `Simulate_Ca_video.m` including `int_var`, `get_out_path`, `create_baseline`, `add_remapping_drifting`, `make_xy_motion_ar`, and `apply_xy_translation`
- `[⚠️ FIXED] S2.C-02` `Add_NRmotion.m`
- `[⚠️ FIXED] S2.C-03` `create_neuron_data.m` including `create_spatial`, `conv_spikes`, and `fix_inactive`
- `[⚠️ FIXED] S2.C-04` `create_random_spike.m` including `create_sine`
- `[⚠️ FIXED] S2.C-05` `BL_sim.m`
- `[⚠️ FIXED] S2.C-06` `estimate_time_constant_PV.m`
- `[✅ DONE] S2.C-07` `bin_data.m`
- `[⚠️ FIXED] S2.C-08` `create_heart_mask.m`
- `[⚠️ FIXED] S2.C-09` `get_random_points.m`
- `[✅ DONE] S2.C-10` `saveash5.m`
- `[⚠️ FIXED] S2.C-11` `save_as_avi.m`
- `[⚠️ FIXED] S2.C-12` `v2uint8.m`
- `[⚠️ FIXED] S2.C-13` `sim_drifting_activities.m`
- `[✅ DONE] S2.C-14` `ar2exp.m`
- `[⚠️ FIXED] S2.C-15` `add_paths.m`
- `[⚠️ FIXED] S2.C-16` CaliAli `vesselness_PV.m`
- `[⚠️ FIXED] S2.C-17` CaliAli `vesselness2D.m`
- `[✅ DONE] S2.C-18` CaliAli `videoConvert.m`
- `[⚠️ FIXED] S2.C-19` CaliAli `estimate_time_constant.m`
- `[⚠️ FIXED] S2.C-20` CaliAli `GetSn.m`
- `[✅ DONE] S2.X1` Replace repo progress usage with `progress.py` over `tqdm` and exclude vendored `ProgressBar.m` from 1:1 parity scope

### Packaging and Validation

- `[✅ DONE] S2.P1` Package Python code with `pyproject.toml`, `__init__.py`, module docstrings, and type hints
- `[✅ DONE] S2.P2` Install into the dedicated `uv` environment and verify `import simulate_ca_imaging_video`
- `[✅ DONE] S2.P3` Gate: full Python codebase runs without MATLAB runtime
- `[✅ DONE] S2.P4` Run the full Python test suite end to end
- `[✅ DONE] S2.P5` Gate: all conversion and parity tests pass with documented tolerances

### Visualization and Reporting

- `[✅ DONE] S2.V1` Create a Jupyter notebook reproducing the MATLAB visualizations
- `[✅ DONE] S2.V2` Label each visualization with its source MATLAB file/function
- `[✅ DONE] S2.V3` Include MATLAB vs Python comparisons where possible using saved parity artifacts
- `[✅ DONE] S2.V4` Gate: notebook runs cleanly and renders all figures
- `[✅ DONE] S2.R1` Write a module-by-module conversion status report
- `[✅ DONE] S2.R2` Include pass/fail counts and max deviation per module
- `[✅ DONE] S2.R3` Flag toolbox-dependent functions and Python workarounds
- `[✅ DONE] S2.R4` Final gate: reprint the full Stage 1 plus Stage 2 RAPHL list with final statuses
