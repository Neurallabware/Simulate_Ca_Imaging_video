from __future__ import annotations

import numpy as np
from scipy.io import loadmat

from simulate_ca_imaging_video import BL_sim
from simulate_ca_imaging_video._compat import get_data_path


def test_s2_c05_bl_sim_matches_matlab_with_score_tolerance(matlab_engine_session):
    mr = loadmat(get_data_path("BL.mat"), squeeze_me=False, struct_as_record=False)["Mr"]
    py_in = [mr[idx, 0] for idx in range(3)]
    py_scores = BL_sim(py_in)
    matlab_engine_session.eval("load('BL.mat','Mr'); s = BL_sim(Mr(1:3,1));", nargout=0)
    matlab_scores = np.array(matlab_engine_session.workspace["s"]).ravel()
    assert np.max(np.abs(py_scores - matlab_scores)) < 0.055
    assert np.mean(np.abs(py_scores - matlab_scores)) < 0.05


def test_s2_c05_bl_sim_smoke_contract():
    mr = loadmat(get_data_path("BL.mat"), squeeze_me=False, struct_as_record=False)["Mr"]
    py_scores = BL_sim([mr[0, 0]])
    assert py_scores.shape == (1,)
    assert np.isfinite(py_scores).all()
