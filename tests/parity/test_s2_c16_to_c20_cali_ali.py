from __future__ import annotations

import matlab
import numpy as np
from scipy.io import loadmat

from simulate_ca_imaging_video.cali_ali_mirror.estimate_time_constant import estimate_time_constant
from simulate_ca_imaging_video.cali_ali_mirror.get_sn import get_sn
from simulate_ca_imaging_video.cali_ali_mirror.video_convert import video_convert
from simulate_ca_imaging_video.cali_ali_mirror.vesselness2d import vesselness2d
from simulate_ca_imaging_video.cali_ali_mirror.vesselness_pv import vesselness_pv
from simulate_ca_imaging_video._compat import get_data_path

from conftest import matlab_to_numpy


def test_s2_c16_vesselness_pv_matches_matlab_with_tolerance(matlab_engine_session):
    img = loadmat(get_data_path("BL.mat"), squeeze_me=False, struct_as_record=False)["Mr"][0, 0][:, :, 0]
    py_out = vesselness_pv(img, 0, np.arange(0.5, 2.0 + 1e-9, 0.5), 0)
    matlab_engine_session.workspace["img"] = matlab.double(img.tolist())
    matlab_engine_session.eval("vpv = vesselness_PV(img, 0, 0.5:0.5:2, 0);", nargout=0)
    matlab_out = matlab_to_numpy(matlab_engine_session.workspace["vpv"])
    assert np.max(np.abs(py_out - matlab_out)) < 0.15
    assert np.mean(np.abs(py_out - matlab_out)) < 0.02


def test_s2_c17_vesselness2d_matches_matlab_with_tolerance(matlab_engine_session):
    img = loadmat(get_data_path("BL.mat"), squeeze_me=False, struct_as_record=False)["Mr"][0, 0][:, :, 0]
    img = (img - img.min()) / (img.max() - img.min())
    py_out = vesselness2d(img, np.arange(0.5, 2.0 + 1e-9, 0.5), (1.0, 1.0), 0.8, False, 0)
    matlab_engine_session.workspace["img"] = matlab.double(img.tolist())
    matlab_engine_session.eval("v2 = vesselness2D(img, 0.5:0.5:2, [1;1], 0.8, false, 0);", nargout=0)
    matlab_out = matlab_to_numpy(matlab_engine_session.workspace["v2"])
    assert np.max(np.abs(py_out - matlab_out)) < 0.15
    assert np.mean(np.abs(py_out - matlab_out)) < 0.02


def test_s2_c18_video_convert_roundtrip_and_matlab_match(matlab_engine_session):
    arr = np.arange(1, 1 + 4 * 5 * 3, dtype=np.float64).reshape(4, 5, 3, order="F")
    py_out = video_convert(video_convert(arr))
    matlab_engine_session.workspace["arr"] = matlab.double(arr[:, :, 0].tolist())
    matlab_engine_session.eval("clear arr3; arr3(:,:,1)=arr;", nargout=0)
    matlab_engine_session.workspace["arr"] = matlab.double(arr[:, :, 1].tolist())
    matlab_engine_session.eval("arr3(:,:,2)=arr;", nargout=0)
    matlab_engine_session.workspace["arr"] = matlab.double(arr[:, :, 2].tolist())
    matlab_engine_session.eval("arr3(:,:,3)=arr; arr4 = videoConvert(videoConvert(arr3));", nargout=0)
    matlab_out = matlab_to_numpy(matlab_engine_session.workspace["arr4"])
    assert np.array_equal(py_out, arr)
    assert np.array_equal(py_out, matlab_out)


def test_s2_c19_estimate_time_constant_matches_matlab(matlab_engine_session):
    y = np.sin(np.linspace(0, 4 * np.pi, 200)) + 0.05 * np.random.RandomState(1).randn(200)
    py_out = estimate_time_constant(y, 2)
    matlab_engine_session.workspace["y"] = matlab.double([y.tolist()])
    matlab_engine_session.eval("g = estimate_time_constant(y, 2);", nargout=0)
    matlab_out = matlab_to_numpy(matlab_engine_session.workspace["g"]).ravel()
    assert np.allclose(py_out, matlab_out, atol=0.1, rtol=0.1)


def test_s2_c20_get_sn_matches_matlab(matlab_engine_session):
    y = np.sin(np.linspace(0, 4 * np.pi, 200)) + 0.05 * np.random.RandomState(1).randn(200)
    py_out = np.asarray(get_sn(y)).ravel()
    matlab_engine_session.workspace["y"] = matlab.double([y.tolist()])
    matlab_engine_session.eval("sn = GetSn(y);", nargout=0)
    matlab_out = np.asarray(matlab_to_numpy(matlab_engine_session.workspace["sn"])).ravel()
    assert np.allclose(py_out, matlab_out, atol=0.02, rtol=0.1)
