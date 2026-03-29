from __future__ import annotations

from pathlib import Path

import matlab
import matlab.engine
import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def matlab_to_numpy(value):
    if isinstance(value, matlab.double):
        return np.array(value, dtype=np.float64)
    if isinstance(value, dict):
        return {key: matlab_to_numpy(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(matlab_to_numpy(item) for item in value)
    return value


def to_matlab_vector(values) -> matlab.double:
    arr = np.asarray(values, dtype=np.float64).ravel()
    return matlab.double([arr.tolist()])


@pytest.fixture(scope="session")
def matlab_engine_session():
    eng = matlab.engine.start_matlab("-nojvm")
    eng.cd(str(REPO_ROOT), nargout=0)
    eng.addpath(str(REPO_ROOT / "sim_matlab" / "scripts"), nargout=0)
    eng.setup_repro_paths(str(REPO_ROOT), nargout=0)
    yield eng
    eng.quit()
