"""Generate the heart-shaped support mask used for neuron placement."""

from __future__ import annotations

import numpy as np
from skimage.draw import polygon2mask


def create_heart_mask(d: tuple[int, int] | list[int] | np.ndarray) -> np.ndarray:
    d1, d2 = int(d[0]), int(d[1])
    t = np.linspace(-np.pi, np.pi, 350)
    ratio = np.ones_like(t)
    nz = t != 0
    ratio[nz] = np.sin(t[nz]) / t[nz]
    x = t * np.sin(np.pi * 0.9 * ratio)
    y = -np.abs(t) * np.cos(np.pi * ratio)
    x = x - x.min()
    x = x / x.max()
    y = y - y.min()
    y = y / y.max()
    x = x * d2 * 0.6 + d2 * 0.2
    y = y * d1 * 0.6 + d1 * 0.2
    return polygon2mask((d1, d2), np.column_stack([y, x]))
