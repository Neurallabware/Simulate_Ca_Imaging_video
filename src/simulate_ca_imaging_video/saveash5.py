"""HDF5 stack writing utilities matching the MATLAB saveash5 helper."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np


def saveash5(data: np.ndarray, path_to_file: str, options: dict[str, Any] | None = None) -> Path:
    options = dict(options or {})
    groupname = options.get("groupname", "/Object")
    append = bool(options.get("append", True))
    path = Path(path_to_file)
    path = path.with_suffix(".h5")
    array = np.asarray(data)
    dataset_name = groupname[1:] if groupname.startswith("/") else groupname
    if not path.exists() or not append:
        mode = "w"
    else:
        mode = "a"
    with h5py.File(path, mode) as handle:
        if dataset_name in handle and not append:
            del handle[dataset_name]
        if dataset_name not in handle:
            maxshape = list(array.shape)
            if array.ndim >= 3:
                maxshape[2] = None
                chunks = (array.shape[0], array.shape[1], 1)
            else:
                maxshape[-1] = None
                chunks = True
            dataset = handle.create_dataset(
                dataset_name,
                data=array,
                maxshape=tuple(maxshape),
                chunks=chunks,
                dtype=array.dtype,
            )
        else:
            dataset = handle[dataset_name]
            start = dataset.shape[2]
            dataset.resize(start + array.shape[2], axis=2)
            dataset[:, :, start : start + array.shape[2]] = array
    return path
