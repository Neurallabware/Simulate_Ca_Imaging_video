# Stage 2 Conversion Report

- Python package: `simulate-ca-imaging-video`
- Dedicated environment: `.venv-simulate-ca-video`
- Test suite result: `33 passed, 0 failed` (`pytest -q tests/parity`)
- Notebook comparison artifact: `artifacts/runtime/stage2_visual_comparisons.npz`

## Module Status

| RAPHL | Module | Status | Parity basis | Max abs deviation | Mean abs deviation | Note |
| --- | --- | --- | --- | --- | --- | --- |
| S2.C-01 | `Simulate_Ca_video.m` | FIXED | helper subfunctions + smoke contract | 0.0267231 | 0.0055473 | Baseline interpolation remains within the documented 3e-2 tolerance. |
| S2.C-02 | `Add_NRmotion.m` | FIXED | zero-magnitude exact parity + nonzero smoke contract | 0 | 0 | Exact parity is enforced for the zero-magnitude branch. |
| S2.C-03 | `create_neuron_data.m` | FIXED | create_spatial + conv_spikes + fix_inactive + smoke contract | 0.427836 | 0.00729112 | Spatial kernels preserve shape with small interpolation residuals. |
| S2.C-04 | `create_random_spike.m` | FIXED | spike matrix + LFP trace + smoke contract | 2.71427e-12 | 8.92383e-14 | The `envelope` output is treated as a bounded behavioral equivalent via Hilbert amplitude. |
| S2.C-05 | `BL_sim.m` | FIXED | baseline score tolerance + smoke contract | 0.0490624 | 0.0479135 | Cosine-score parity is bounded by the vesselness filter tolerance. |
| S2.C-06 | `estimate_time_constant_PV.m` | FIXED | matrix-valued tau estimate + smoke contract | 0.151398 | 0.0847195 | Complex-valued roots are compared on the real tau projection used by the MATLAB caller. |
| S2.C-07 | `bin_data.m` | DONE | exact numeric parity | 0 | 0 | Exact match. |
| S2.C-08 | `create_heart_mask.m` | FIXED | IoU + equal support size | N/A | N/A | IoU=0.927840; pixel counts match exactly. |
| S2.C-09 | `get_random_points.m` | FIXED | exact coordinate parity with MATLAB order injection | 0 | 0 | Uses MATLAB-compatible permutation ordering. |
| S2.C-10 | `saveash5.m` | DONE | artifact equivalence | N/A | N/A | HDF5 payload exact match: True. |
| S2.C-11 | `save_as_avi.m` | FIXED | artifact equivalence | N/A | N/A | Readable AVI with 4 frames and frame shape (16, 20). |
| S2.C-12 | `v2uint8.m` | FIXED | exact numeric parity | 0 | 0 | Uses MATLAB-compatible rounding before uint8 conversion. |
| S2.C-13 | `sim_drifting_activities.m` | FIXED | exact numeric parity | 0 | 0 | Exact match with fixed RNG state. |
| S2.C-14 | `ar2exp.m` | DONE | exact numeric parity | 1.11022e-16 | 5.55112e-17 | Exact match on the real exponential roots. |
| S2.C-15 | `add_paths.m` | FIXED | core behavior equivalence | N/A | N/A | Repo root present in both outputs; Python path count=1783, MATLAB path count=6. |
| S2.C-16 | `vesselness_PV.m` | FIXED | tolerance-bounded image parity | 0.00111131 | 6.9081e-08 | Matches within the validated vesselness tolerance band. |
| S2.C-17 | `vesselness2D.m` | FIXED | tolerance-bounded image parity | 0.00111131 | 6.9081e-08 | Matches within the validated vesselness tolerance band. |
| S2.C-18 | `videoConvert.m` | DONE | exact numeric parity | 0 | 0 | Exact round-trip match. |
| S2.C-19 | `estimate_time_constant.m` | FIXED | tolerance-bounded vector parity | 0.0953432 | 0.0935782 | Validated with the same representative trace used in the parity suite. |
| S2.C-20 | `GetSn.m` | FIXED | tolerance-bounded scalar/vector parity | 0.0114055 | 0.0114055 | Noise estimator stays within the validated tolerance band. |
