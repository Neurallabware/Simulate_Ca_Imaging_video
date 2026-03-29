from __future__ import annotations

import matlab
import numpy as np

from simulate_ca_imaging_video.create_random_spike import create_random_spike, create_sine

from conftest import matlab_to_numpy, to_matlab_vector


def test_s2_c04_create_random_spike_no_lfp_matches_matlab_with_fixed_uniforms(matlab_engine_session):
    n = 3
    f = 8
    sf = 1
    p = np.array([0.2, 0.4, 0.1], dtype=np.float64)
    u = np.array(
        [
            [0.1, 0.8, 0.3, 0.9, 0.05, 0.7, 0.8, 0.2],
            [0.9, 0.2, 0.1, 0.6, 0.7, 0.3, 0.4, 0.8],
            [0.05, 0.9, 0.95, 0.85, 0.15, 0.25, 0.35, 0.45],
        ],
        dtype=np.float64,
    )
    py_s, py_lfp = create_random_spike(
        n,
        f,
        sf,
        p,
        0,
        1,
        0,
        spike_uniform=u,
    )
    matlab_s, matlab_lfp = matlab_engine_session.parity_create_random_spike_helpers(
        "create_random_spike_no_lfp",
        float(n),
        float(f),
        float(sf),
        to_matlab_vector(p),
        0.0,
        matlab.double(u.tolist()),
        to_matlab_vector([1, 1, 1]),
        nargout=2,
    )
    matlab_s = matlab_to_numpy(matlab_s)
    matlab_lfp = matlab_to_numpy(matlab_lfp)
    assert np.allclose(py_s, matlab_s, atol=1e-8, rtol=0)
    assert np.allclose(np.asarray(py_lfp).reshape(1, -1), matlab_lfp, atol=1e-8, rtol=0)


def test_s2_c04_create_sine_matches_trace_and_approximate_envelope(matlab_engine_session):
    lfp = 8
    sf = 10
    f = 40
    phase_offsets = np.linspace(-1.2, 1.2, int(round((lfp + 5 - (lfp - 4)) / 0.01)) + 1)
    py_lfp, py_fp = create_sine(lfp, sf, f, phase_offsets=phase_offsets)
    matlab_lfp, matlab_fp = matlab_engine_session.parity_create_random_spike_helpers(
        "create_sine",
        float(lfp),
        float(sf),
        float(f),
        to_matlab_vector(phase_offsets),
        nargout=2,
    )
    matlab_lfp = matlab_to_numpy(matlab_lfp).ravel()
    assert np.allclose(py_lfp, matlab_lfp, atol=1e-8, rtol=0)
    assert np.min(py_fp) >= 0
    assert np.max(py_fp) <= 1.05
    assert py_fp.shape == matlab_to_numpy(matlab_fp).ravel().shape


def test_s2_c04_create_random_spike_lfp_smoke_contract():
    p = np.array([0.2, 0.4, 0.1], dtype=np.float64)
    s, lfp = create_random_spike(3, 20, 10, p, 8, 1, 1)
    assert s.shape[0] == 3
    assert s.shape[1] == 20
    assert lfp.ndim == 1
    assert lfp.size == 240
