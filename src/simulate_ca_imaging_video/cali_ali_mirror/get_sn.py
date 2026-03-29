"""Python mirror of CaliAli's GetSn MATLAB helper."""

from __future__ import annotations

import numpy as np
from scipy.signal import welch


def get_sn(
    y: np.ndarray,
    range_ff: tuple[float, float] = (0.25, 0.5),
    method: str = "logmexp",
) -> np.ndarray:
    arr = np.asarray(y, dtype=np.float64)
    if 1 in arr.shape:
        arr = arr.reshape(-1, 1)
    else:
        arr = arr.T
    arr[~np.isfinite(arr)] = 0
    ff, psdx = welch(arr, fs=1.0, axis=0, nperseg=min(256, arr.shape[0]))
    psdx = np.asarray(psdx)
    if psdx.ndim == 1:
        psdx = psdx[:, np.newaxis]
    keep = (ff >= range_ff[0]) & (ff <= range_ff[1])
    selected = psdx[keep, :] / 2
    if method == "mean":
        sn = np.sqrt(np.mean(selected, axis=0))
    elif method == "median":
        sn = np.sqrt(np.median(selected, axis=0))
    else:
        sn = np.sqrt(np.exp(np.mean(np.log(selected), axis=0)))
    return np.asarray(sn)


GetSn = get_sn
