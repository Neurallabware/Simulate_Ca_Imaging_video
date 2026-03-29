"""MATLAB-compatible float-to-uint8 conversion utilities."""

from __future__ import annotations

import numpy as np


def v2uint8(values: np.ndarray, thr: tuple[float, float] | None = None) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if thr is not None:
        arr = np.clip(arr, thr[0], thr[1])
    arr_min = np.nanmin(arr)
    arr_max = np.nanmax(arr)
    if not np.isfinite(arr_min) or not np.isfinite(arr_max) or arr_max <= arr_min:
        scaled = np.zeros_like(arr, dtype=np.float64)
    else:
        scaled = (arr - arr_min) / (arr_max - arr_min)
    scaled = np.clip(np.rint(scaled * (2**8)), 0, 255)
    return scaled.astype(np.uint8)
