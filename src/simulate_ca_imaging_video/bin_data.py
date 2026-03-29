"""Temporal binning helpers used by the calcium simulation pipeline."""

from __future__ import annotations

import math

import numpy as np


def bin_data(a: np.ndarray, sf: float, binsize: float) -> np.ndarray:
    data = np.asarray(a)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    k = int(round(binsize * sf))
    if k <= 0:
        raise ValueError("binsize * sf must be positive.")
    n_blocks = int(math.ceil(data.shape[1] / k))
    out = np.zeros((data.shape[0], n_blocks), dtype=data.dtype)
    for block_index in range(n_blocks):
        start = block_index * k
        stop = min(start + k, data.shape[1])
        out[:, block_index] = data[:, start:stop].sum(axis=1)
    return out
