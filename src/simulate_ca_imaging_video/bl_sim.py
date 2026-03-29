"""Baseline scoring utilities ported from the MATLAB BL_sim workflow."""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist
from skimage.exposure import equalize_adapthist

from .cali_ali_mirror.vesselness_pv import vesselness_pv


def bl_sim(mr: list[np.ndarray] | np.ndarray) -> np.ndarray:
    seq = mr if isinstance(mr, list) else list(mr)
    scores = np.zeros(len(seq), dtype=np.float64)
    for idx, item in enumerate(seq):
        x = np.asarray(item)
        vf = np.zeros_like(x, dtype=np.float64)
        for frame_idx in range(2):
            filtered = vesselness_pv(x[:, :, frame_idx], 0, np.arange(0.6, 0.9 + 1e-12, 0.032) * 2.5)
            vf[:, :, frame_idx] = equalize_adapthist(filtered)
        scores[idx] = 1 - pdist(vf.reshape((-1, vf.shape[2]), order="F").T, "cosine")[0]
    return scores


BL_sim = bl_sim
