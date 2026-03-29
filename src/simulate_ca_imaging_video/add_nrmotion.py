"""Non-rigid across-session motion utilities."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

from ._compat import crop_edges, mat2gray


def _warp_with_displacement(values: np.ndarray, displacement: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    rows, cols = np.meshgrid(
        np.arange(arr.shape[0], dtype=np.float64),
        np.arange(arr.shape[1], dtype=np.float64),
        indexing="ij",
    )
    row_coords = rows - displacement[:, :, 0]
    col_coords = cols - displacement[:, :, 1]
    coords = np.vstack([row_coords.ravel(), col_coords.ravel()])
    if arr.ndim == 2:
        warped = map_coordinates(arr, coords, order=1, mode="constant", cval=0.0)
        return warped.reshape(arr.shape)
    frames = []
    for idx in range(arr.shape[2]):
        warped = map_coordinates(arr[:, :, idx], coords, order=1, mode="constant", cval=0.0)
        frames.append(warped.reshape(arr.shape[:2]))
    return np.stack(frames, axis=2)


def add_nrmotion(
    m: np.ndarray,
    a: np.ndarray,
    ba: np.ndarray,
    magnitude: float = 2,
    plotme: int = 0,
    translation_misalignment: float = 0,
    msz: float = 60,
    *,
    rng: np.random.RandomState | None = None,
    displacement_field: np.ndarray | None = None,
    translation_vector: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    del plotme
    rng = np.random.RandomState() if rng is None else rng
    if displacement_field is None:
        d1, d2 = np.asarray(m).shape[:2]
        noise = rng.randn(d1, d2)
        smooth = gaussian_filter(noise, sigma=msz, mode="wrap", truncate=max(4.0, 1000 / max(msz, 1)))
        img1 = mat2gray(smooth)
        gy, gx = np.gradient(img1)
        if translation_misalignment > 0:
            t = rng.randn(2) if translation_vector is None else np.asarray(translation_vector, dtype=np.float64)
            norm = np.sqrt(np.sum(t**2))
            if norm == 0:
                norm = 1.0
            t = t * (translation_misalignment / norm)
            gx = (mat2gray(gx) - 0.5) * 2 * magnitude + t[0]
            gy = (mat2gray(gy) - 0.5) * 2 * magnitude + t[1]
        else:
            gx = (mat2gray(gx) - 0.5) * 2 * magnitude
            gy = (mat2gray(gy) - 0.5) * 2 * magnitude
        displacement_field = np.stack([gy, gx], axis=2)
    dm = crop_edges(_warp_with_displacement(m, displacement_field))
    da = crop_edges(_warp_with_displacement(a, displacement_field))
    dba = crop_edges(_warp_with_displacement(ba, displacement_field))
    d = crop_edges(displacement_field)
    return dm, da, dba, d


Add_NRmotion = add_nrmotion
