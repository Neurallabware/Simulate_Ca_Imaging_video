"""Thin tqdm-backed replacement for the vendored MATLAB progress bar."""

from __future__ import annotations

from typing import Iterable, TypeVar

from tqdm.auto import tqdm

T = TypeVar("T")


def progress(iterable: Iterable[T], enabled: bool = True, **kwargs: object) -> Iterable[T]:
    if not enabled:
        return iterable
    return tqdm(iterable, **kwargs)
