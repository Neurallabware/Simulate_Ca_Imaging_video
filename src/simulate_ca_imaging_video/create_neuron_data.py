"""Neuron footprint and calcium activity synthesis for simulated sessions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve2d
from skimage.transform import resize

from ._compat import ensure_rng, load_repo_variable, matlab_rand, matlab_randperm
from .create_heart_mask import create_heart_mask
from .create_random_spike import create_random_spike
from .get_random_points import get_random_points


def _opt_get(opt: Mapping[str, Any], key: str) -> Any:
    return opt[key]


def _matlab_find_coords(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coords = np.argwhere(np.asarray(mask).T)
    x = coords[:, 0]
    y = coords[:, 1]
    return x, y


def create_spatial(
    bw: np.ndarray,
    opt: Mapping[str, Any],
    *,
    rng: np.random.RandomState | None = None,
    points: tuple[np.ndarray, np.ndarray] | None = None,
    shape_indices: np.ndarray | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    rng = ensure_rng(rng)
    x, y = _matlab_find_coords(bw)
    if points is None:
        x_points, y_points = get_random_points(
            x,
            y,
            _opt_get(opt, "min_dist"),
            _opt_get(opt, "Nneu"),
            _opt_get(opt, "plotme"),
            rng=rng,
        )
    else:
        x_points, y_points = points
    if _opt_get(opt, "CA1_A"):
        a = load_repo_variable("A_CA1.mat", "A")
    elif _opt_get(opt, "A2"):
        a = load_repo_variable("A2.mat", "A")
    else:
        a = load_repo_variable("A.mat", "A")
    n = x_points.size
    if a.ndim == 4 and a.shape[3] > 1:
        if shape_indices is None:
            ix = matlab_randperm(rng, a.shape[3], n)
        else:
            ix = np.asarray(shape_indices, dtype=int)
        weights = np.linspace(0, 1, _opt_get(opt, "ses"))
        as_cube = np.zeros((a.shape[0], a.shape[1], n, _opt_get(opt, "ses")), dtype=np.float64)
        for idx, weight in enumerate(weights):
            as_cube[:, :, :, idx] = a[:, :, 0, ix] * (1 - weight) + a[:, :, 1, ix] * weight
    else:
        if shape_indices is None:
            ix = np.floor(matlab_rand(rng, n) * a.shape[2]).astype(int)
        else:
            ix = np.asarray(shape_indices, dtype=int)
        as_cube = np.repeat(a[:, :, ix][:, :, :, np.newaxis], _opt_get(opt, "ses"), axis=3)
    if _opt_get(opt, "scale") > 1:
        d1, d2, s, c = as_cube.shape
        resized = np.zeros(
            (round(d1 * _opt_get(opt, "scale")), round(d2 * _opt_get(opt, "scale")), s, c),
            dtype=as_cube.dtype,
        )
        for si in range(s):
            for ci in range(c):
                resized[:, :, si, ci] = resize(
                    as_cube[:, :, si, ci],
                    (resized.shape[0], resized.shape[1]),
                    order=3,
                    preserve_range=True,
                    anti_aliasing=False,
                )
        as_cube = resized
    a_out: list[np.ndarray] = []
    b_out: list[np.ndarray] = []
    for session in range(_opt_get(opt, "ses")):
        a_temp = np.zeros((_opt_get(opt, "d")[0], _opt_get(opt, "d")[1], n), dtype=np.float64)
        b_temp = np.zeros_like(a_temp)
        for idx in range(n):
            z = np.zeros(_opt_get(opt, "d"), dtype=np.float64)
            z[int(y_points[idx]), int(x_points[idx])] = 1
            a_temp[:, :, idx] = convolve2d(z, as_cube[:, :, idx, session], mode="same")
            b_temp[:, :, idx] = gaussian_filter(
                a_temp[:, :, idx],
                sigma=10,
                mode="nearest",
                truncate=max(4.0, _opt_get(opt, "d")[1] / 20),
            )
        if np.max(a_temp) > 0:
            a_temp = a_temp / np.max(a_temp)
        if np.max(b_temp) > 0:
            b_temp = b_temp / np.max(b_temp)
        a_out.append(a_temp)
        b_out.append(b_temp)
    return a_out, b_out


def conv_spikes(
    s: np.ndarray,
    invt_rise: np.ndarray,
    invt_decay: np.ndarray,
    f: int,
    *,
    rng: np.random.RandomState | None = None,
    rise_draws: np.ndarray | None = None,
    decay_draws: np.ndarray | None = None,
) -> np.ndarray:
    rng = ensure_rng(rng)
    time = np.arange(f)
    out = np.zeros_like(s, dtype=np.float64)
    if rise_draws is None:
        rise_draws = rng.lognormal(invt_rise[0], invt_rise[1], size=s.shape[0])
    if decay_draws is None:
        decay_draws = rng.lognormal(invt_decay[0], invt_decay[1], size=s.shape[0])
    for idx in range(s.shape[0]):
        rise = 1 / rise_draws[idx]
        decay = 1 / decay_draws[idx]
        g = np.exp(-time / decay) - np.exp(-time / rise)
        c = np.convolve(s[idx], g)
        out[idx, :] = c[:f]
    return out


def fix_inactive(s: np.ndarray, *, rng: np.random.RandomState | None = None) -> np.ndarray:
    rng = ensure_rng(rng)
    out = np.array(s, copy=True)
    d = out.shape[1]
    inactive = np.where(np.sum(out, axis=1) == 0)[0]
    for idx in inactive:
        out[idx, int(rng.randint(d))] = 1
    return out


def create_neuron_data(
    opt: Mapping[str, Any],
    *,
    rng: np.random.RandomState | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    rng = ensure_rng(rng if rng is not None else _opt_get(opt, "seed"))
    if _opt_get(opt, "create_mask"):
        bw = create_heart_mask(_opt_get(opt, "d"))
    else:
        bw = np.ones(_opt_get(opt, "d"), dtype=bool)
    a_out, ba = create_spatial(bw, opt, rng=rng)
    n = a_out[0].shape[2]
    spike_prob = np.asarray(_opt_get(opt, "spike_prob"), dtype=np.float64).copy()
    spike_prob[0] = np.log(np.exp(spike_prob[0]) / _opt_get(opt, "sf"))
    spike_prob[1] = spike_prob[1] / _opt_get(opt, "sf")
    invt_rise = np.asarray(_opt_get(opt, "invtRise"), dtype=np.float64).copy()
    invt_rise[0] = np.log(np.exp(invt_rise[0]) / _opt_get(opt, "sf"))
    invt_decay = np.asarray(_opt_get(opt, "invtDecay"), dtype=np.float64).copy()
    invt_decay[0] = np.log(np.exp(invt_decay[0]) / _opt_get(opt, "sf"))
    p = rng.lognormal(spike_prob[0], spike_prob[1], size=(n,))
    lfp_list = []
    s_list = []
    for _ in range(_opt_get(opt, "ses")):
        spikes, lfp = create_random_spike(
            n,
            _opt_get(opt, "F"),
            _opt_get(opt, "sf"),
            p,
            _opt_get(opt, "LFP"),
            1,
            _opt_get(opt, "force_active"),
            rng=rng,
        )
        lfp_list.append(np.asarray(lfp).reshape(1, -1))
        s_list.append(np.asarray(spikes, dtype=np.float64))
    s = np.concatenate(s_list, axis=1)
    lfp = np.concatenate(lfp_list, axis=1) if lfp_list else np.empty((1, 0))
    s = fix_inactive(s, rng=rng)
    c = conv_spikes(s, invt_rise, invt_decay, s.shape[1], rng=rng)
    return a_out, ba, c, s, lfp
