# Calcium video simulator

Provides the retained MATLAB calcium-video simulator and an active Python port with motion and ground-truth metadata.

## Runtime and prerequisites

The Python package declares Python >=3.11 and installs from pyproject.toml. Bundled footprint/baseline MAT assets are under data/.

## Install the Python port

Use Python >=3.11 and install the lab checkout:

```bash
python -m pip install -e .
```

The source assets under `data/` provide the bundled footprint/baseline MAT files.

## Python example

```python
from pathlib import Path
from simulate_ca_imaging_video import simulate_ca_video

output = Path("outputs/demo")
output.mkdir(parents=True, exist_ok=True)

paths, out_dir, motion, stem, footprints, gt_motion_xy = simulate_ca_video(
    Nneu=10, F=64, ses=2, seed=42,
    outpath=str(output), save_avi=False,
)
```

These are illustrative source-supported options, not a runtime test from this documentation rollout. `save_avi=False` selects the HDF5 path. Set output controls explicitly if generating data for a benchmark.

## MATLAB reference

From the repository root in MATLAB:

```matlab
addpath(genpath(fullfile(pwd, 'sim_matlab')));
[Mot, outpath, file_name, A, GT_motion_xy] = Simulate_Ca_video('Nneu', 100, 'ses', 2);
```

The Python function returns six items, including output paths as its first item. The MATLAB example returns five values in a different order.

## Inputs, outputs, and handoff

The Python simulate_ca_video function returns six items: output paths, output directory, motion list, file-name stem, footprint list, and GT_motion_xy. Internal movies use (H, W, T).

The Python return order is (paths, output_directory, motion, file_name, footprints, gt_motion_xy). Movie construction uses (H, W, T) and Fortran-order spatial flattening. Saved metadata includes A_GT, A, C, S, BL, opt, Mot, LFP, and GT_motion_xy. Keep session/ROI identities and clean/noisy/ground-truth pairings intact.

## Version-specific notes

Set the random seed, session count, frame count, motion options, and output directory explicitly. MATLAB and Python return orders differ; do not copy MATLAB tuple unpacking into Python.

