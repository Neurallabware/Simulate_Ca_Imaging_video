"""Python mirror of CaliAli's vesselness_PV MATLAB helper."""

from __future__ import annotations

import numpy as np

from .._compat import mat2gray
from .video_convert import video_convert
from .vesselness2d import vesselness2d


def _apply_vesselness_filter(image: np.ndarray, sz: np.ndarray, norm: int) -> np.ndarray:
    ip = mat2gray(np.asarray(image, dtype=np.float64))
    return vesselness2d(ip, sz, (1.0, 1.0), 0.8, False, norm)


def vesselness_pv(
    video: np.ndarray,
    use_parallel: int = 1,
    sz: np.ndarray | list[float] | None = None,
    norm: int = 0,
) -> np.ndarray:
    del use_parallel
    scales = np.asarray(sz if sz is not None else np.arange(0.5, 2.0 + 1e-9, 0.5), dtype=np.float64)
    if np.asarray(video).ndim == 2:
        return _apply_vesselness_filter(video, scales, norm)
    frames = video_convert(np.asarray(video))
    filtered = [_apply_vesselness_filter(frame, scales, norm) for frame in frames]
    return video_convert(filtered)


vesselness_PV = vesselness_pv
