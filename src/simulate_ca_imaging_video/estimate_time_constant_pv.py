"""Population-vector time-constant estimation routines."""

from __future__ import annotations

import numpy as np

from .ar2exp import ar2exp
from .cali_ali_mirror.estimate_time_constant import estimate_time_constant


def estimate_time_constant_pv(c: np.ndarray) -> np.ndarray:
    traces = np.asarray(c, dtype=np.float64)
    tau = np.zeros((traces.shape[0], 2), dtype=np.complex128)
    for idx in range(traces.shape[0]):
        tau[idx, :] = ar2exp(estimate_time_constant(traces[idx, :], 2))
    return tau


estimate_time_constant_PV = estimate_time_constant_pv
