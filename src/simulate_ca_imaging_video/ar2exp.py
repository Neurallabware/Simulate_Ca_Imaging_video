"""Convert AR(2) coefficients into exponential time constants."""

from __future__ import annotations

import numpy as np


def ar2exp(g: np.ndarray | list[float]) -> np.ndarray:
    coeffs = np.asarray(g, dtype=np.complex128).ravel()
    if coeffs.size == 1:
        coeffs = np.array([coeffs[0], 0], dtype=np.complex128)
    roots = np.roots([1, -coeffs[0], -coeffs[1]])
    d = roots[np.argmax(roots.real)]
    r = roots[np.argmin(roots.real)]
    tau_d = -1 / np.log(d)
    tau_r = -1 / np.log(r)
    return np.asarray([tau_d, tau_r])
