# Stage 1 Failure Log

| Asset | Initial Failure | Fix | Re-run Result |
| --- | --- | --- | --- |
| `BL_sim` | `vesselness_PV` was unresolved on the default MATLAB path. | Added exact CaliAli vesselness paths in `setup_repro_paths.m`. | `PASS` under clean launcher; final status `FIXED`. |
| `estimate_time_constant_PV` | `estimate_time_constant` was unresolved on the default MATLAB path. | Added exact OASIS function path in `setup_repro_paths.m`. | `PASS` under clean launcher; final status `FIXED`. |
