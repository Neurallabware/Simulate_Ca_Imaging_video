from __future__ import annotations

import matlab
import numpy as np

from simulate_ca_imaging_video.simulate_ca_video import (
    add_remapping_drifting,
    apply_xy_translation,
    create_baseline,
    int_var,
    make_xy_motion_ar,
    simulate_ca_video,
)

from conftest import matlab_to_numpy, to_matlab_vector


def test_s2_c01_int_var_matches_matlab(matlab_engine_session):
    kwargs = dict(
        Nneu=5,
        ses=3,
        F=20,
        save_videos=False,
        save_mat=False,
        save_GT=False,
        B=1,
        seed=1,
        batch_sz=7,
    )
    py_opt = int_var(**kwargs)
    matlab_opt = matlab_engine_session.parity_simulate_ca_video_helpers(
        "int_var",
        "Nneu",
        5.0,
        "ses",
        3.0,
        "F",
        20.0,
        "save_videos",
        0.0,
        "save_mat",
        0.0,
        "save_GT",
        0.0,
        "B",
        1.0,
        "seed",
        1.0,
        "batch_sz",
        7.0,
        nargout=1,
    )
    matlab_opt = matlab_to_numpy(matlab_opt)
    assert py_opt["Nneu"] == matlab_opt["Nneu"]
    assert py_opt["ses"] == matlab_opt["ses"]
    assert py_opt["F"] == matlab_opt["F"]
    assert py_opt["batch_sz"] == matlab_opt["batch_sz"]
    assert py_opt["save_videos"] == matlab_opt["save_videos"]
    assert py_opt["save_mat"] == matlab_opt["save_mat"]
    assert py_opt["save_GT"] == matlab_opt["save_GT"]
    assert py_opt["seed"] == matlab_opt["seed"]


def test_s2_c01_create_baseline_matches_matlab(matlab_engine_session):
    kwargs = dict(d=[80, 100], ses=2, B=1, seed=1, save_videos=False, save_mat=False, save_GT=False)
    py_opt = int_var(**kwargs)
    py_bl = create_baseline(py_opt)
    matlab_bl = matlab_engine_session.parity_simulate_ca_video_helpers(
        "create_baseline",
        "d",
        to_matlab_vector([80, 100]),
        "ses",
        2.0,
        "B",
        1.0,
        "seed",
        1.0,
        "save_videos",
        0.0,
        "save_mat",
        0.0,
        "save_GT",
        0.0,
        nargout=1,
    )
    matlab_bl = matlab_to_numpy(matlab_bl)
    assert np.allclose(py_bl, matlab_bl, atol=3e-2, rtol=0)


def test_s2_c01_add_remapping_drifting_matches_matlab(matlab_engine_session):
    c = np.arange(1, 1 + 5 * 12, dtype=np.float64).reshape(5, 12, order="F")
    py_opt = int_var(ses=3, F=4, overlap=0.5, overlapMulti=0.25, seed=1, save_videos=False, save_mat=False, save_GT=False)
    py_opt["Nneu2"] = 5
    py_c = add_remapping_drifting(c, py_opt)
    matlab_c = matlab_engine_session.parity_simulate_ca_video_helpers(
        "add_remapping_drifting",
        matlab.double(c.tolist()),
        "ses",
        3.0,
        "F",
        4.0,
        "overlap",
        0.5,
        "overlapMulti",
        0.25,
        "seed",
        1.0,
        "save_videos",
        0.0,
        "save_mat",
        0.0,
        "save_GT",
        0.0,
        "Nneu2",
        5.0,
        nargout=1,
    )
    matlab_c = matlab_to_numpy(matlab_c)
    assert np.allclose(py_c, matlab_c, atol=1e-8, rtol=0)


def test_s2_c01_make_xy_motion_ar_matches_matlab_with_fixed_innovations(matlab_engine_session):
    phi = 0.9
    max_px = 2.0
    e1 = np.array([0.1, 0.2, -0.3, 0.4], dtype=np.float64)
    e2 = np.array([0.2, -0.1, 0.0, 0.1], dtype=np.float64)
    py_dx, py_dy = make_xy_motion_ar(4, phi=phi, std_px=1.0, max_px=max_px, seed=1, innovations=(e1, e2))
    matlab_dx, matlab_dy = matlab_engine_session.parity_simulate_ca_video_helpers(
        "make_xy_motion_ar_from_innov",
        phi,
        max_px,
        to_matlab_vector(e1),
        to_matlab_vector(e2),
        nargout=2,
    )
    matlab_dx = matlab_to_numpy(matlab_dx).ravel()
    matlab_dy = matlab_to_numpy(matlab_dy).ravel()
    assert np.allclose(py_dx, matlab_dx, atol=1e-8, rtol=0)
    assert np.allclose(py_dy, matlab_dy, atol=1e-8, rtol=0)


def test_s2_c01_apply_xy_translation_matches_matlab(matlab_engine_session):
    y = np.arange(1, 1 + 8 * 9 * 3, dtype=np.float64).reshape(8, 9, 3, order="F")
    dx = np.array([0.3, -0.5, 1.2], dtype=np.float64)
    dy = np.array([0.1, 0.4, -0.2], dtype=np.float64)
    py_out = apply_xy_translation(y, dx, dy)
    matlab_engine_session.workspace["Y"] = matlab.double(y[:, :, 0].tolist())
    matlab_engine_session.eval("clear Y3; Y3(:,:,1) = Y;", nargout=0)
    matlab_engine_session.workspace["Y"] = matlab.double(y[:, :, 1].tolist())
    matlab_engine_session.eval("Y3(:,:,2) = Y;", nargout=0)
    matlab_engine_session.workspace["Y"] = matlab.double(y[:, :, 2].tolist())
    matlab_engine_session.eval("Y3(:,:,3) = Y;", nargout=0)
    matlab_out = matlab_engine_session.parity_simulate_ca_video_helpers(
        "apply_xy_translation",
        matlab_engine_session.workspace["Y3"],
        to_matlab_vector(dx),
        to_matlab_vector(dy),
        nargout=1,
    )
    matlab_out = matlab_to_numpy(matlab_out)
    assert np.allclose(py_out, matlab_out, atol=1e-6, rtol=0)


def test_s2_c01_simulate_ca_video_smoke_contract():
    pth_out, outpath, mot, file_name, a, gt_motion_xy = simulate_ca_video(
        Nneu=4,
        ses=1,
        F=10,
        save_videos=False,
        save_mat=False,
        save_GT=False,
        session_motion_std=0,
        seed=1,
    )
    assert pth_out == [None]
    assert outpath is None
    assert file_name is None
    assert len(mot) == 1
    assert len(a) == 1
    assert len(gt_motion_xy) == 1
    assert a[0].shape[0] > 0 and a[0].shape[1] > 0 and a[0].shape[2] == 4
