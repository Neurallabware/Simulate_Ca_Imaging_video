"""Random point selection with MATLAB-compatible ordering semantics."""

from __future__ import annotations

import numpy as np

from ._compat import matlab_randperm


def get_random_points(
    x: np.ndarray,
    y: np.ndarray,
    min_allowable_distance: float,
    number_of_points: int,
    plotme: bool = False,
    rng: np.random.RandomState | None = None,
    order: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    del plotme
    rng = np.random.RandomState() if rng is None else rng
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    order = matlab_randperm(rng, len(y)) if order is None else np.asarray(order, dtype=int)
    x = x[order]
    y = y[order]
    xs = [x[0]]
    ys = [y[0]]
    count = 2
    for idx in range(1, len(x)):
        this_x = x[idx]
        this_y = y[idx]
        distances = np.sqrt((this_x - np.asarray(xs)) ** 2 + (this_y - np.asarray(ys)) ** 2)
        if float(distances.min()) >= min_allowable_distance:
            xs.append(this_x)
            ys.append(this_y)
            count += 1
        if count > number_of_points:
            break
    return np.asarray(xs), np.asarray(ys)
