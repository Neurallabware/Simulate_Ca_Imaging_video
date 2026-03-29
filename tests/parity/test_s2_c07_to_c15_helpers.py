from __future__ import annotations

from pathlib import Path

import h5py
import imageio.v2 as imageio
import matlab
import numpy as np

from simulate_ca_imaging_video import (
    add_paths,
    ar2exp,
    bin_data,
    create_heart_mask,
    get_random_points,
    save_as_avi,
    saveash5,
    sim_drifting_activities,
    v2uint8,
)

from conftest import matlab_to_numpy, to_matlab_vector


def test_s2_c07_bin_data_matches_matlab(matlab_engine_session):
    a = np.arange(1, 13, dtype=np.float64).reshape(2, 6, order="F")
    py_out = bin_data(a, 1, 2)
    matlab_out = matlab_engine_session.bin_data(matlab.double(a.tolist()), 1.0, 2.0, nargout=1)
    matlab_out = matlab_to_numpy(matlab_out)
    assert np.allclose(py_out, matlab_out, atol=1e-8, rtol=0)


def test_s2_c08_create_heart_mask_matches_matlab(matlab_engine_session):
    py_out = create_heart_mask((80, 100))
    matlab_out = matlab_engine_session.create_heart_mask(to_matlab_vector([80, 100]), nargout=1)
    matlab_out = matlab_to_numpy(matlab_out)
    py_mask = py_out.astype(bool)
    matlab_mask = np.asarray(matlab_out).astype(bool)
    intersection = np.logical_and(py_mask, matlab_mask).sum()
    union = np.logical_or(py_mask, matlab_mask).sum()
    assert intersection / union > 0.92
    assert py_mask.sum() == matlab_mask.sum()


def test_s2_c09_get_random_points_matches_matlab(matlab_engine_session):
    x = np.array([10, 20, 30, 40, 50, 60], dtype=np.float64)
    y = np.array([10, 20, 30, 40, 50, 60], dtype=np.float64)
    matlab_engine_session.eval("rng(1,'twister');", nargout=0)
    matlab_order = matlab_engine_session.randperm(6.0, 6.0, nargout=1)
    matlab_order = matlab_to_numpy(matlab_order).ravel().astype(int) - 1
    py_x, py_y = get_random_points(x, y, 5, 4, rng=np.random.RandomState(1), order=matlab_order)
    matlab_engine_session.eval("rng(1,'twister');", nargout=0)
    matlab_x, matlab_y = matlab_engine_session.get_random_points(
        to_matlab_vector(x),
        to_matlab_vector(y),
        5.0,
        4.0,
        0.0,
        nargout=2,
    )
    matlab_x = matlab_to_numpy(matlab_x).ravel()
    matlab_y = matlab_to_numpy(matlab_y).ravel()
    assert np.array_equal(py_x, matlab_x)
    assert np.array_equal(py_y, matlab_y)


def test_s2_c10_saveash5_appends_expected_dataset(tmp_path):
    data1 = np.arange(1, 1 + 4 * 5 * 2, dtype=np.uint8).reshape(4, 5, 2, order="F")
    data2 = np.arange(1, 1 + 4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3, order="F")
    out = tmp_path / "demo_stack"
    saveash5(data1, str(out))
    saveash5(data2, str(out), {"append": True})
    with h5py.File(out.with_suffix(".h5"), "r") as handle:
        dataset = handle["Object"][:]
    expected = np.concatenate([data1, data2], axis=2)
    assert np.array_equal(dataset, expected)


def test_s2_c11_save_as_avi_writes_readable_video(tmp_path):
    video = np.arange(1, 1 + 16 * 20 * 4, dtype=np.uint8).reshape(16, 20, 4, order="F")
    path = save_as_avi(video, str(tmp_path / "demo.avi"))
    assert path.exists()
    reader = imageio.get_reader(path)
    frames = [reader.get_data(i) for i in range(4)]
    reader.close()
    assert len(frames) == 4
    assert frames[0].shape[0] == 16
    assert frames[0].shape[1] == 20


def test_s2_c12_v2uint8_matches_matlab(matlab_engine_session):
    values = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float64)
    py_out = v2uint8(values)
    matlab_out = matlab_engine_session.v2uint8(matlab.double(values.tolist()), nargout=1)
    matlab_out = matlab_to_numpy(matlab_out)
    assert np.array_equal(py_out.astype(np.float64), matlab_out)


def test_s2_c13_sim_drifting_activities_matches_matlab(matlab_engine_session):
    py_out = sim_drifting_activities(12, 6, rng=np.random.RandomState(1))
    matlab_engine_session.eval("rng(1,'twister');", nargout=0)
    matlab_out = matlab_engine_session.sim_drifting_activities(12.0, 6.0, nargout=1)
    matlab_out = matlab_to_numpy(matlab_out)
    assert np.allclose(py_out, matlab_out, atol=1e-8, rtol=0)


def test_s2_c14_ar2exp_matches_matlab(matlab_engine_session):
    g = np.array([0.8, -0.1], dtype=np.float64)
    py_out = ar2exp(g)
    matlab_out = matlab_engine_session.ar2exp(to_matlab_vector(g), nargout=1)
    matlab_out = matlab_to_numpy(matlab_out).ravel()
    assert np.allclose(np.real(py_out), np.real(matlab_out), atol=1e-8, rtol=0)


def test_s2_c15_add_paths_matches_core_behavior(matlab_engine_session):
    py_paths = add_paths()
    matlab_paths = matlab_engine_session.add_paths(nargout=1)
    matlab_paths = list(matlab_paths)
    repo_root = str(Path.cwd())
    assert repo_root in py_paths
    assert repo_root in matlab_paths
    assert all(".git" not in path for path in py_paths)
