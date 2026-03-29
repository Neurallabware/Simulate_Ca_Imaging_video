"""Compatibility helpers that reproduce MATLAB array and RNG behavior."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.io import loadmat


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = Path("/mnt/nas02/Dataset/Simulate_Ca_Imaging_video")


def get_repo_root() -> Path:
    return REPO_ROOT


def get_data_root() -> Path:
    repo_data = REPO_ROOT / "data"
    if repo_data.is_dir():
        return repo_data
    return REPO_ROOT


def get_data_path(filename: str | Path) -> Path:
    filename = Path(filename)
    candidate = get_data_root() / filename
    if candidate.exists():
        return candidate
    fallback = REPO_ROOT / filename
    if fallback.exists():
        return fallback
    dataset_fallback = DEFAULT_DATA_ROOT / filename
    if dataset_fallback.exists():
        return dataset_fallback
    return candidate


def matlab_kwargs(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    if args:
        if len(args) != 1 or not isinstance(args[0], dict):
            raise TypeError("Expected keyword arguments or a single options dict.")
        merged = dict(args[0])
        merged.update(kwargs)
        return merged
    return dict(kwargs)


def ensure_rng(seed: Any = None) -> np.random.RandomState:
    if isinstance(seed, np.random.RandomState):
        return seed
    if seed is None or seed == "shuffle":
        seed = int(np.random.SeedSequence().generate_state(1)[0])
    return np.random.RandomState(int(seed))


def matlab_rand(rng: np.random.RandomState, shape: int | tuple[int, ...]) -> np.ndarray:
    if isinstance(shape, int):
        return rng.rand(shape)
    size = int(np.prod(shape))
    return rng.rand(size).reshape(shape, order="F")


def matlab_randn(rng: np.random.RandomState, shape: int | tuple[int, ...]) -> np.ndarray:
    if isinstance(shape, int):
        return rng.randn(shape)
    size = int(np.prod(shape))
    return rng.randn(size).reshape(shape, order="F")


def matlab_randperm(rng: np.random.RandomState, n: int, k: int | None = None) -> np.ndarray:
    if k is None:
        k = n
    available = list(range(n))
    out: list[int] = []
    for _ in range(k):
        pick = int(np.floor(rng.rand() * len(available)))
        out.append(available.pop(pick))
    return np.asarray(out, dtype=int)


def matlab_hdf_to_numpy(dataset: h5py.Dataset) -> np.ndarray:
    arr = np.array(dataset)
    if arr.ndim >= 2:
        arr = np.transpose(arr, tuple(reversed(range(arr.ndim))))
    return arr


def load_mat_variable(path: str | Path, variable: str) -> Any:
    path = Path(path)
    try:
        data = loadmat(path, squeeze_me=False, struct_as_record=False)
        if variable not in data:
            raise KeyError(f"{variable!r} not found in {path}")
        return data[variable]
    except NotImplementedError:
        with h5py.File(path, "r") as handle:
            if variable not in handle:
                raise KeyError(f"{variable!r} not found in {path}")
            return matlab_hdf_to_numpy(handle[variable])


def load_repo_variable(filename: str, variable: str) -> Any:
    path = get_data_path(filename)
    return load_mat_variable(path, variable)


def mat2gray(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    lo = np.nanmin(arr)
    hi = np.nanmax(arr)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(arr, dtype=np.float64)
    return (arr - lo) / (hi - lo)


def rescale(values: np.ndarray, out_min: float = 0.0, out_max: float = 1.0) -> np.ndarray:
    arr = mat2gray(values)
    return arr * (out_max - out_min) + out_min


def crop_edges(values: np.ndarray, margin: int = 20) -> np.ndarray:
    if margin <= 0:
        return values
    return values[margin:-margin, margin:-margin, ...]


def flatten_frames(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values)
    spatial = arr.shape[:2]
    trailing = int(np.prod(arr.shape[2:])) if arr.ndim > 2 else 1
    return arr.reshape((spatial[0] * spatial[1], trailing), order="F")
