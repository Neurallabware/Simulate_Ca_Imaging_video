"""Python mirror of CaliAli's vesselness2D MATLAB helper."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import convolve1d

from .._compat import mat2gray


def _gradient2(values: np.ndarray, option: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    out = np.zeros_like(arr)
    k, l = arr.shape
    if option.lower() == "x":
        out[0, :] = arr[1, :] - arr[0, :]
        out[k - 1, :] = arr[k - 1, :] - arr[k - 2, :]
        out[1 : k - 1, :] = (arr[2:k, :] - arr[0 : k - 2, :]) / 2
    elif option.lower() == "y":
        out[:, 0] = arr[:, 1] - arr[:, 0]
        out[:, l - 1] = arr[:, l - 1] - arr[:, l - 2]
        out[:, 1 : l - 1] = (arr[:, 2:l] - arr[:, 0 : l - 2]) / 2
    else:
        raise ValueError(f"Unknown option {option!r}")
    return out


def _imgaussian(values: np.ndarray, sigma: float, spacing: tuple[float, float], siz: float | None = None) -> np.ndarray:
    if siz is None:
        siz = sigma * 6
    if sigma <= 0:
        return np.asarray(values, dtype=np.float32)
    x = np.arange(-np.ceil(siz / spacing[0] / 2), np.ceil(siz / spacing[0] / 2) + 1)
    hx = np.exp(-(x**2 / (2 * (sigma / spacing[0]) ** 2)))
    hx = hx / hx.sum()
    y = np.arange(-np.ceil(siz / spacing[1] / 2), np.ceil(siz / spacing[1] / 2) + 1)
    hy = np.exp(-(y**2 / (2 * (sigma / spacing[1]) ** 2)))
    hy = hy / hy.sum()
    filtered = convolve1d(np.asarray(values, dtype=np.float32), hx.astype(np.float32), axis=0, mode="nearest")
    filtered = convolve1d(filtered, hy.astype(np.float32), axis=1, mode="nearest")
    return filtered


def _hessian2d(values: np.ndarray, sigma: float, spacing: tuple[float, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    filtered = _imgaussian(values, sigma, spacing) if sigma > 0 else np.asarray(values, dtype=np.float32)
    dy = _gradient2(filtered, "y")
    dyy = _gradient2(dy, "y")
    dx = _gradient2(filtered, "x")
    dxx = _gradient2(dx, "x")
    dxy = _gradient2(dx, "y")
    return dxx, dyy, dxy


def _eigval_of_hessian_2d(dxx: np.ndarray, dxy: np.ndarray, dyy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tmp = np.sqrt((dxx - dyy) ** 2 + 4 * dxy**2)
    mu1 = 0.5 * (dxx + dyy + tmp)
    mu2 = 0.5 * (dxx + dyy - tmp)
    check = np.abs(mu1) > np.abs(mu2)
    lambda1 = mu1.copy()
    lambda2 = mu2.copy()
    lambda1[check] = mu2[check]
    lambda2[check] = mu1[check]
    return lambda1, lambda2


def _image_eigenvalues(
    values: np.ndarray,
    sigma: float,
    spacing: tuple[float, float],
    brightondark: bool,
) -> tuple[np.ndarray, np.ndarray]:
    hxx, hyy, hxy = _hessian2d(values, sigma, spacing)
    c = sigma**2
    hxx = c * hxx
    hxy = c * hxy
    hyy = c * hyy
    b1 = -(hxx + hyy)
    b2 = hxx * hyy - hxy**2
    mask = np.ones_like(b1, dtype=bool)
    if brightondark:
        mask[b1 < 0] = False
        mask[(b2 == 0) & (b1 == 0)] = False
    else:
        mask[b1 > 0] = False
        mask[(b2 == 0) & (b1 == 0)] = False
    lambda1 = np.zeros_like(b1, dtype=np.float32)
    lambda2 = np.zeros_like(b1, dtype=np.float32)
    l1, l2 = _eigval_of_hessian_2d(hxx[mask], hxy[mask], hyy[mask])
    lambda1[mask] = l1
    lambda2[mask] = l2
    lambda1[~np.isfinite(lambda1)] = 0
    lambda2[~np.isfinite(lambda2)] = 0
    lambda1[np.abs(lambda1) < 1e-4] = 0
    lambda2[np.abs(lambda2) < 1e-4] = 0
    return lambda1, lambda2


def vesselness2d(
    image: np.ndarray,
    sigmas: np.ndarray | list[float],
    spacing: tuple[float, float] | list[float],
    tau: float,
    brightondark: bool = False,
    norm: int = 0,
) -> np.ndarray:
    del tau
    arr = np.asarray(image, dtype=np.float32)
    out = np.zeros(arr.shape, dtype=np.float32)
    for sigma in np.asarray(sigmas).ravel():
        _, lambda2 = _image_eigenvalues(arr, float(sigma), tuple(spacing), brightondark)
        if brightondark:
            lambda2 = -lambda2
        if norm == 1:
            out = np.maximum(out, mat2gray(lambda2).astype(np.float32))
        elif norm == 0:
            out = out + lambda2
        elif norm == 2:
            out = np.maximum(out, lambda2)
        else:
            raise ValueError(f"Unsupported norm mode {norm}")
    return out


vesselness2D = vesselness2d
