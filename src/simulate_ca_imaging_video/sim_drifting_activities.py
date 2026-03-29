"""Generate session-to-session drift modulation factors."""

from __future__ import annotations

import math

import numpy as np

from ._compat import matlab_rand


def sim_drifting_activities(
    n: int,
    ses: int,
    *,
    rng: np.random.RandomState | None = None,
) -> np.ndarray:
    rng = np.random.RandomState() if rng is None else rng
    half_ses = ses // 2
    ns1 = int(math.ceil(n * 0.25))
    ns2 = int(math.ceil(n * 0.25))
    a1 = np.concatenate([np.zeros(ns2), np.ones(ns1), np.zeros(ns1 + ns2)])
    a2 = np.concatenate([np.ones(ns2), np.zeros(ns1), np.zeros(ns1 + ns2)])
    a = np.column_stack([a1, a2])
    out = np.vstack([np.linspace(row[0], row[1], half_ses) for row in a])
    a = out
    if n > a.shape[0]:
        a = np.vstack([a, matlab_rand(rng, (n - a.shape[0], half_ses))])
    b1 = np.concatenate([np.zeros(ns1 + ns2), np.zeros(ns2), np.ones(ns1)])
    b2 = np.concatenate([np.zeros(ns1 + ns2), np.ones(ns2), np.zeros(ns1)])
    b = np.column_stack([b1, b2])
    out = np.vstack([np.linspace(row[0], row[1], half_ses) for row in b])
    b = out
    if n > b.shape[0]:
        b = np.vstack([b, matlab_rand(rng, (n - b.shape[0], half_ses))])
    return np.hstack([a, b])
