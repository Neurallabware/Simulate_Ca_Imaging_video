from __future__ import annotations

import matlab
import numpy as np

from simulate_ca_imaging_video import estimate_time_constant_PV


def test_s2_c06_estimate_time_constant_pv_matches_matlab(matlab_engine_session):
    c = np.vstack(
        [
            np.linspace(0.1, 1.0, 50),
            np.cos(np.linspace(0, 3 * np.pi, 50)) + 0.2 * np.sin(np.linspace(0, 10, 50)),
        ]
    )
    py_tau = estimate_time_constant_PV(c)
    matlab_tau = matlab_engine_session.estimate_time_constant_PV(matlab.double(c.tolist()), nargout=1)
    matlab_tau = np.array(matlab_tau)
    assert np.allclose(np.real(py_tau), np.real(matlab_tau), atol=0.25, rtol=0.05)


def test_s2_c06_estimate_time_constant_pv_smoke_contract():
    c = np.vstack(
        [
            np.linspace(0.1, 1.0, 50),
            np.cos(np.linspace(0, 3 * np.pi, 50)) + 0.2 * np.sin(np.linspace(0, 10, 50)),
        ]
    )
    tau = estimate_time_constant_PV(c)
    assert tau.shape == (2, 2)
    assert np.isfinite(np.real(tau)).all()
