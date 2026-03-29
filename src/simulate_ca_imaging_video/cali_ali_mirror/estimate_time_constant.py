"""Python mirror of CaliAli's estimate_time_constant MATLAB helper."""

from __future__ import annotations

import numpy as np
from scipy.linalg import toeplitz

from .get_sn import get_sn


def _biased_xcov(y: np.ndarray, lags: int) -> np.ndarray:
    arr = np.asarray(y, dtype=np.float64).ravel()
    arr = arr - np.mean(arr)
    xc = np.empty(lags + 1, dtype=np.float64)
    for k in range(lags + 1):
        xc[k] = arr[k:] @ arr[: arr.size - k]
    xc = np.concatenate([xc[1:][::-1], xc]) / arr.size
    return xc


def estimate_time_constant(
    y: np.ndarray,
    p: int = 2,
    sn: float | np.ndarray | None = None,
    lags: int = 5,
    fudge_factor: float = 1.0,
) -> np.ndarray | float:
    if sn is None or (isinstance(sn, np.ndarray) and sn.size == 0):
        sn = get_sn(y)
    lags = lags + p
    xc = _biased_xcov(np.asarray(y, dtype=np.float64), lags).reshape(-1, 1)
    a = toeplitz(xc[lags : lags + lags].ravel(), xc[lags : lags + p].ravel()) - np.asarray(sn).item() ** 2 * np.eye(lags, p)
    g = np.linalg.pinv(a) @ xc[lags + 1 :]
    coeffs = g.ravel()
    while np.max(np.abs(np.roots(np.concatenate([[1.0], -coeffs])))) > 1 and p < 5:
        p += 1
        return estimate_time_constant(y, p, sn, lags, fudge_factor)
    if p == 5 and np.max(np.abs(np.roots(np.concatenate([[1.0], -coeffs])))) > 1:
        return 0.0
    rg = np.roots(np.concatenate([[1.0], -coeffs]))
    if not np.isrealobj(rg):
        rg = np.real(rg) + 0.001 * np.random.randn(*rg.shape)
    rg = np.real(rg)
    mask_hi = rg > 1
    rg[mask_hi] = 0.95 + 0.001 * np.random.randn(mask_hi.sum())
    mask_lo = rg < 0
    rg[mask_lo] = 0.15 + 0.001 * np.random.randn(mask_lo.sum())
    pg = np.poly(fudge_factor * rg)
    return -pg[1:]
