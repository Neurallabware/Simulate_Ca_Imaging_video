from __future__ import annotations

import matlab
import numpy as np

from simulate_ca_imaging_video import create_neuron_data
from simulate_ca_imaging_video.simulate_ca_video import int_var
from simulate_ca_imaging_video.create_neuron_data import conv_spikes, create_spatial, fix_inactive

from conftest import matlab_to_numpy, to_matlab_vector


def test_s2_c03_create_spatial_matches_matlab_with_fixed_points_and_indices(matlab_engine_session):
    bw = np.ones((80, 90), dtype=np.float64)
    py_opt = int_var(d=[80, 90], ses=2, scale=1, save_videos=False, save_mat=False, save_GT=False)
    py_points = (np.array([29, 59]), np.array([19, 39]))
    py_ix = np.array([0, 10])
    py_aout, py_ba = create_spatial(bw, py_opt, points=py_points, shape_indices=py_ix)
    matlab_aout, matlab_ba = matlab_engine_session.parity_create_neuron_data_helpers(
        "create_spatial",
        matlab.double(bw.tolist()),
        to_matlab_vector([30, 60]),
        to_matlab_vector([20, 40]),
        to_matlab_vector([1, 11]),
        "d",
        to_matlab_vector([80, 90]),
        "ses",
        2.0,
        "scale",
        1.0,
        "seed",
        1.0,
        nargout=2,
    )
    matlab_aout = matlab_to_numpy(matlab_aout)
    matlab_ba = matlab_to_numpy(matlab_ba)
    for idx in range(2):
        assert np.max(np.abs(py_aout[idx] - matlab_aout[idx])) < 0.45
        assert np.mean(np.abs(py_aout[idx] - matlab_aout[idx])) < 0.01
        assert np.max(np.abs(py_ba[idx] - matlab_ba[idx])) < 0.09
        assert np.mean(np.abs(py_ba[idx] - matlab_ba[idx])) < 0.01


def test_s2_c03_conv_spikes_matches_matlab_with_fixed_draws(matlab_engine_session):
    s = np.array([[1, 0, 1, 0, 0, 1], [0, 1, 0, 0, 1, 0]], dtype=np.float64)
    invt_rise = np.array([2.08, 0.29], dtype=np.float64)
    invt_decay = np.array([0.55, 0.44], dtype=np.float64)
    rise_draws = np.array([2.0, 3.0], dtype=np.float64)
    decay_draws = np.array([1.5, 2.5], dtype=np.float64)
    py_out = conv_spikes(s, invt_rise, invt_decay, s.shape[1], rise_draws=rise_draws, decay_draws=decay_draws)
    matlab_out = matlab_engine_session.parity_create_neuron_data_helpers(
        "conv_spikes",
        matlab.double(s.tolist()),
        to_matlab_vector(invt_rise),
        to_matlab_vector(invt_decay),
        to_matlab_vector(rise_draws),
        to_matlab_vector(decay_draws),
        float(s.shape[1]),
        nargout=1,
    )
    matlab_out = matlab_to_numpy(matlab_out)
    assert np.allclose(py_out, matlab_out, atol=1e-8, rtol=0)


def test_s2_c03_fix_inactive_identity_on_active_input(matlab_engine_session):
    s = np.array([[1, 0, 0], [0, 1, 1]], dtype=np.float64)
    py_out = fix_inactive(s)
    matlab_out = matlab_engine_session.parity_create_neuron_data_helpers(
        "fix_inactive",
        matlab.double(s.tolist()),
        nargout=1,
    )
    matlab_out = matlab_to_numpy(matlab_out)
    assert np.array_equal(py_out, s)
    assert np.array_equal(matlab_out, s)
    assert np.array_equal(py_out, matlab_out)


def test_s2_c03_create_neuron_data_smoke_contract():
    opt = int_var(Nneu=4, ses=2, F=20, save_videos=False, save_mat=False, save_GT=False, session_motion_std=0, seed=1)
    a, ba, c, s, lfp = create_neuron_data(opt)
    assert len(a) == 2
    assert len(ba) == 2
    assert a[0].shape[2] == 4
    assert ba[0].shape == a[0].shape
    assert c.shape == (4, 40)
    assert s.shape == (4, 40)
    assert lfp.shape == (1, 40)
