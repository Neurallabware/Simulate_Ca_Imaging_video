# Inputs, outputs, and provenance

The Python simulate_ca_video function returns six items: output paths, output directory, motion list, file-name stem, footprint list, and GT_motion_xy. Internal movies use (H, W, T).

The Python return order is (paths, output_directory, motion, file_name, footprints, gt_motion_xy). Movie construction uses (H, W, T) and Fortran-order spatial flattening. Saved metadata includes A_GT, A, C, S, BL, opt, Mot, LFP, and GT_motion_xy. Keep session/ROI identities and clean/noisy/ground-truth pairings intact.

Record seed, geometry, temporal sampling, motion, noise, and clean/noisy/ground-truth correspondence. A configured simulator does not establish that generated data match an experimental distribution.
