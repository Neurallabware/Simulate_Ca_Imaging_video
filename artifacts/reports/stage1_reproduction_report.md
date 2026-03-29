# Stage 1 Reproduction Report

## Environment
- MATLAB binary: `/home/yz/MATLAB/R2025b/bin/matlab`
- Required toolboxes: MATLAB, Image Processing Toolbox, Signal Processing Toolbox, Statistics and Machine Learning Toolbox, Computer Vision Toolbox
- GPU availability checked: 8 devices available, but CPU execution path is valid and used for Stage 1 verification

## Data
- Primary data policy: repo-bundled `.mat` files are the source of truth
- Data manifest: `artifacts/reports/data_manifest.csv`
- No external download URLs or login-gated assets were required by the repo during reproduction

## Executable Asset Summary
- PASS: 16
- FIXED: 2
- UNRESOLVED: 0
- Test suites: `N/A` (none present)
- Notebooks/live scripts: `N/A` (none present)

## Notes
- `BL_sim` required the exact CaliAli vesselness implementation on the MATLAB path
- `estimate_time_constant_PV` required the exact CaliAli/OASIS `estimate_time_constant` implementation on the MATLAB path
- Both path-related failures were fixed through the clean launcher, without modifying original repo source files

## Verdict
- `FULLY REPRODUCED`
