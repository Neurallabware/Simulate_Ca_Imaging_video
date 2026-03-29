"""AVI export helpers that replace MATLAB VideoWriter behavior."""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np


def save_as_avi(video: np.ndarray, video_file_name: str) -> Path:
    path = Path(video_file_name)
    if path.suffix.lower() != ".avi":
        path = path.with_suffix(".avi")
    frames = np.asarray(video)
    with imageio.get_writer(
        path,
        format="FFMPEG",
        fps=25,
        codec="ffv1",
        macro_block_size=None,
    ) as writer:
        for idx in range(frames.shape[2]):
            writer.append_data(np.asarray(frames[:, :, idx]))
    return path
