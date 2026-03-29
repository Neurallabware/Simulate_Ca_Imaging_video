"""Python mirror of CaliAli's videoConvert MATLAB helper."""

from __future__ import annotations

import numpy as np


def video_convert(data: np.ndarray | list[np.ndarray]) -> np.ndarray | list[np.ndarray]:
    if isinstance(data, np.ndarray) and data.ndim == 3:
        return [data[:, :, idx] for idx in range(data.shape[2])]
    if isinstance(data, list) and data and all(np.asarray(item).ndim == 2 for item in data):
        return np.stack([np.asarray(item) for item in data], axis=2)
    raise ValueError("Input must be a 3D array or a list of 2D arrays.")


videoConvert = video_convert
