from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import savemat

from simulate_ca_imaging_video import Add_NRmotion

from conftest import matlab_to_numpy


def _matlab_quote(path: Path) -> str:
    return str(path).replace("'", "''")


def test_s2_c02_add_nrmotion_zero_magnitude_matches_matlab(matlab_engine_session, tmp_path):
    m = np.arange(1, 1 + 64 * 72, dtype=np.float64).reshape(64, 72, order="F")
    a = np.stack([m / 10, m / 20], axis=2)
    ba = np.stack([m / 30, m / 40], axis=2)
    payload = tmp_path / "add_nrmotion_fixture.mat"
    savemat(payload, {"M": m, "A": a, "bA": ba})
    matlab_engine_session.eval(f"s = load('{_matlab_quote(payload)}');", nargout=0)
    matlab_engine_session.eval(
        "[dM, dA, dbA, D] = Add_NRmotion(s.M, s.A, s.bA, 0, 0, 0, 10);",
        nargout=0,
    )
    matlab_dm = matlab_to_numpy(matlab_engine_session.workspace["dM"])
    matlab_da = matlab_to_numpy(matlab_engine_session.workspace["dA"])
    matlab_dba = matlab_to_numpy(matlab_engine_session.workspace["dbA"])
    matlab_d = matlab_to_numpy(matlab_engine_session.workspace["D"])
    py_dm, py_da, py_dba, py_d = Add_NRmotion(m, a, ba, 0, 0, 0, 10)
    assert np.allclose(py_dm, matlab_dm, atol=1e-8, rtol=0)
    assert np.allclose(py_da, matlab_da, atol=1e-8, rtol=0)
    assert np.allclose(py_dba, matlab_dba, atol=1e-8, rtol=0)
    assert np.allclose(py_d, matlab_d, atol=1e-8, rtol=0)


def test_s2_c02_add_nrmotion_nonzero_smoke_contract():
    m = np.arange(1, 1 + 64 * 72, dtype=np.float64).reshape(64, 72, order="F")
    a = np.stack([m / 10, m / 20], axis=2)
    ba = np.stack([m / 30, m / 40], axis=2)
    dm, da, dba, d = Add_NRmotion(m, a, ba, 2, 0, 1, 10)
    assert dm.shape == (24, 32)
    assert da.shape == (24, 32, 2)
    assert dba.shape == (24, 32, 2)
    assert d.shape == (24, 32, 2)
    assert np.isfinite(dm).all()
    assert np.isfinite(da).all()
    assert np.isfinite(dba).all()
    assert np.isfinite(d).all()
