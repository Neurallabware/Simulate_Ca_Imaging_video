"""Recursive path discovery that mirrors the MATLAB add_paths helper."""

from __future__ import annotations

from pathlib import Path


def add_paths() -> list[str]:
    repo_root = Path(__file__).resolve().parents[2]
    paths = [repo_root]
    for path in sorted(repo_root.rglob("*")):
        if path.is_dir() and ".git" not in path.parts:
            paths.append(path)
    seen = []
    seen_str = set()
    for path in paths:
        path_str = str(path)
        if path_str not in seen_str:
            seen.append(path_str)
            seen_str.add(path_str)
    return seen
