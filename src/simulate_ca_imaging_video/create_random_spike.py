"""Spike train and LFP synthesis helpers used by the simulator."""

from __future__ import annotations

import math

import numpy as np
from scipy.signal import hilbert

from ._compat import matlab_rand, matlab_randn, matlab_randperm
from .bin_data import bin_data


def create_sine(
    lfp: float,
    sf: float,
    f: int,
    *,
    rng: np.random.RandomState | None = None,
    phase_offsets: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState() if rng is None else rng
    lfp_sf = 120
    freq = np.arange(lfp - 4, lfp + 5.0001, 0.01)
    dt = 1 / lfp_sf
    stop_time = f / lfp_sf
    t = np.arange(dt, stop_time * (lfp_sf / sf) + 1e-12, dt)
    lfp_trace = np.zeros_like(t, dtype=np.float64)
    if phase_offsets is None:
        phase_offsets = matlab_randn(rng, freq.size) * 3
    for idx, current_freq in enumerate(freq):
        lfp_t = np.sin(2 * np.pi * current_freq * t + phase_offsets[idx])
        weight = math.exp(-0.5 * ((current_freq - lfp) / 1.0) ** 2) / math.sqrt(2 * math.pi)
        lfp_trace = lfp_trace + lfp_t * weight
    amp = np.abs(hilbert(lfp_trace))
    yupper = amp
    ylower = -amp
    denom = yupper - ylower
    denom[denom == 0] = 1
    fp = (lfp_trace - ylower) / denom
    return lfp_trace, fp


def create_random_spike(
    n: int,
    f: int,
    sf: float,
    p: np.ndarray,
    lfp: float,
    lfp_strength: float = 1,
    force_active: int = 1,
    *,
    rng: np.random.RandomState | None = None,
    spike_uniform: np.ndarray | None = None,
    phase_offsets: np.ndarray | None = None,
    activation_choices: np.ndarray | None = None,
    shuffle_rows: np.ndarray | None = None,
    shuffle_permutations: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState() if rng is None else rng
    p = np.asarray(p, dtype=np.float64).copy().ravel()
    if lfp > 0:
        p = p * 2
        p[p > 1] = 1
        lfp_signal, lfp_prob = create_sine(lfp, sf, f, rng=rng, phase_offsets=phase_offsets)
    else:
        lfp_prob = np.ones(f, dtype=np.float64)
        lfp_signal = np.ones(f, dtype=np.float64)
    bin_width = lfp_prob.size / f
    if spike_uniform is None:
        spike_uniform = matlab_rand(rng, (n, lfp_prob.size))
    spikes = np.zeros((n, lfp_prob.size), dtype=np.float64)
    activation_choices = np.asarray(activation_choices) if activation_choices is not None else None
    activation_cursor = 0
    for idx in range(n):
        prob = p[idx] / bin_width
        prob = prob * lfp_prob
        temp = (spike_uniform[idx] < prob).astype(np.float64)
        if force_active == 1 and temp.sum() == 0:
            valid = np.flatnonzero(prob > 0)
            if valid.size > 0:
                if activation_choices is not None and activation_cursor < activation_choices.size:
                    choice = int(activation_choices[activation_cursor]) % valid.size
                    activation_cursor += 1
                else:
                    choice = int(np.floor(rng.rand() * valid.size))
                temp[valid[choice]] = 1
        spikes[idx] = temp
    if lfp_strength < 1:
        n_shuffle = int(math.floor((1 - lfp_strength) * n))
        if shuffle_rows is None:
            shuffle_rows = matlab_randperm(rng, n, n_shuffle)
        for cursor, row in enumerate(np.asarray(shuffle_rows, dtype=int)):
            perm = (
                np.asarray(shuffle_permutations[cursor], dtype=int)
                if shuffle_permutations is not None
                else matlab_randperm(rng, spikes.shape[1])
            )
            spikes[row] = spikes[row, perm]
    spikes = bin_data(spikes, 1, bin_width)
    return spikes, lfp_signal
