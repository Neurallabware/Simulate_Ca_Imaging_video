"""Top-level calcium imaging video simulation pipeline."""

from __future__ import annotations

import os
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import savemat
from scipy.ndimage import gaussian_filter, shift

from ._compat import (
    crop_edges,
    ensure_rng,
    flatten_frames,
    load_repo_variable,
    mat2gray,
    matlab_kwargs,
    matlab_rand,
    matlab_randperm,
    rescale,
)
from .add_nrmotion import add_nrmotion
from .create_neuron_data import create_neuron_data
from .progress import progress
from .save_as_avi import save_as_avi
from .saveash5 import saveash5
from .sim_drifting_activities import sim_drifting_activities
from .v2uint8 import v2uint8


DEFAULT_OPTIONS: dict[str, Any] = {
    "outpath": None,
    "min_dist": 8,
    "Nneu": 50,
    "PNR": 2,
    "d": [220, 300],
    "F": 1000,
    "batch_sz": None,
    "overlap": 1,
    "overlapMulti": 0,
    "NR_misalignment": 0,
    "motion_sz": 60,
    "translation_misalignment": 1,
    "session_motion_std": 3,
    "session_motion_phi": 0.99,
    "session_motion_max": None,
    "session_motion_seed": None,
    "ses": 2,
    "seed": "shuffle",
    "B": "1",
    "spike_prob": [-4.9, 2.25],
    "save_videos": True,
    "create_mask": True,
    "comb_fact": 0,
    "drift": 0,
    "LFP": 0,
    "sf": 10,
    "plotme": 0,
    "invtRise": [2.08, 0.29],
    "invtDecay": [0.55, 0.44],
    "disappearBV": 0,
    "CA1_A": False,
    "A2": False,
    "force_active": 1,
    "save_avi": 1,
    "scale": 1,
    "save_mat": True,
    "save_GT": True,
}


def int_var(*args: Any, **kwargs: Any) -> dict[str, Any]:
    values = deepcopy(DEFAULT_OPTIONS)
    values.update(matlab_kwargs(args, kwargs))
    if isinstance(values["B"], str):
        values["B"] = [int(part) for part in values["B"].split()] if " " in values["B"] else [int(values["B"])]
    elif np.isscalar(values["B"]):
        values["B"] = [int(values["B"])]
    values["save_videos"] = bool(values["save_videos"])
    values["save_mat"] = bool(values["save_mat"])
    values["save_GT"] = bool(values["save_GT"])
    if values.get("save_files") is not None and "save_videos" not in kwargs:
        values["save_videos"] = bool(values["save_files"])
    if values["batch_sz"] is None:
        values["batch_sz"] = values["F"]
    values["batch_sz"] = max(1, min(int(values["F"]), int(round(values["batch_sz"]))))
    rng = ensure_rng(values["seed"])
    values["seed"] = int(rng.get_state()[1][0])
    return values


def get_out_path(opt: dict[str, Any]) -> Path | None:
    if opt["save_videos"] or opt["save_mat"] or opt["save_GT"]:
        return Path(opt["outpath"] or os.getcwd()).resolve()
    return None


def create_baseline(opt: dict[str, Any], *, rng: np.random.RandomState | None = None) -> np.ndarray:
    rng = ensure_rng(rng if rng is not None else opt["seed"])
    comb_fact = 1 - opt["comb_fact"]
    ses = opt["ses"]
    mr = load_repo_variable("BL.mat", "Mr")
    b = opt["B"]
    if len(b) == 1:
        selected = np.asarray(mr[b[0] - 1, 0], dtype=np.float64)
        bl = np.zeros((opt["d"][0], opt["d"][1], 2), dtype=np.float64)
        for idx in range(selected.shape[2]):
            from skimage.transform import resize

            bl[:, :, idx] = resize(selected[:, :, idx], tuple(opt["d"]), order=3, preserve_range=True, anti_aliasing=False)
        if ses > 2:
            p = rng.rand(ses)
            temp = bl.copy()
            bl = np.zeros((opt["d"][0], opt["d"][1], ses), dtype=np.float64)
            for idx in range(ses):
                bl[:, :, idx] = temp[:, :, 0] * p[idx] + temp[:, :, 1] * (1 - p[idx]) + rng.randn() / 10
            bl = mat2gray(bl)
    else:
        from skimage.transform import resize

        temp = np.stack(
            [
                resize(np.asarray(mr[b[0] - 1, 0])[:, :, 0], tuple(opt["d"]), order=3, preserve_range=True, anti_aliasing=False),
                resize(np.asarray(mr[b[1] - 1, 0])[:, :, 0], tuple(opt["d"]), order=3, preserve_range=True, anti_aliasing=False),
            ],
            axis=2,
        )
        p = np.linspace(1, comb_fact, ses)
        bl = np.zeros((opt["d"][0], opt["d"][1], ses), dtype=np.float64)
        for idx in range(ses):
            bl[:, :, idx] = temp[:, :, 0] * p[idx] + temp[:, :, 1] * (1 - p[idx])
    if opt["disappearBV"] > 0:
        from scipy.ndimage import median_filter

        for idx in range(bl.shape[2]):
            im = bl[:, :, idx]
            gs = median_filter(np.mean(im, axis=1, keepdims=True) * np.mean(im, axis=0, keepdims=True), size=(25, 25))
            gs = gs + rescale(rng.randn(*im.shape), float(np.min(im)), float(np.max(im))) / 10
            im = rescale(im * (1 - opt["disappearBV"]) + gs * opt["disappearBV"], float(np.min(bl[:, :, idx])), float(np.max(bl[:, :, idx])))
            bl[:, :, idx] = im
    else:
        bl = mat2gray(bl)
    return bl


def add_remapping_drifting(c: np.ndarray, opt: dict[str, Any], *, rng: np.random.RandomState | None = None) -> np.ndarray:
    rng = ensure_rng(rng if rng is not None else opt["seed"])
    nneu2 = opt["Nneu2"]
    ses = opt["ses"]
    f = opt["F"]
    if ses > 1:
        temp = (matlab_rand(rng, (nneu2, ses)) <= opt["overlap"]).astype(np.float64)
        t2 = np.repeat(np.eye(ses), int(np.ceil(nneu2 / ses)), axis=0)
        drop = t2.shape[0] - temp.shape[0]
        if drop > 0:
            remove_ix = matlab_randperm(rng, t2.shape[0], drop)
            keep = np.ones(t2.shape[0], dtype=bool)
            keep[remove_ix] = False
            t2 = t2[keep]
        temp = ((temp + t2) > 0).astype(np.float64)
        m = np.repeat(temp, f, axis=1)
        m[m == 0] = opt["overlapMulti"]
    else:
        m = np.ones((nneu2, f), dtype=np.float64)
    out = c * m
    if opt["drift"] == 1:
        drift = sim_drifting_activities(nneu2, ses, rng=rng)
        drift = np.repeat(drift, f, axis=1)
        out = out * drift
    return out


def make_xy_motion_ar(
    t: int,
    phi: float = 0.95,
    std_px: float = 2.0,
    max_px: float | None = None,
    seed: int | None = None,
    *,
    innovations: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    rng = ensure_rng(seed if seed is not None else 0)
    sigma = std_px * np.sqrt(1 - phi**2)
    if innovations is None:
        e1 = sigma * rng.randn(t)
        e2 = sigma * rng.randn(t)
    else:
        e1, e2 = innovations
    dx = np.zeros(t, dtype=np.float64)
    dy = np.zeros(t, dtype=np.float64)
    for idx in range(t):
        dx[idx] = (phi * dx[idx - 1] if idx > 0 else 0) + e1[idx]
        dy[idx] = (phi * dy[idx - 1] if idx > 0 else 0) + e2[idx]
    dx = dx - dx.mean()
    dy = dy - dy.mean()
    if max_px is not None:
        scale = max(1.0, np.max(np.abs(np.concatenate([dx, dy]))) / max_px)
        dx = dx / scale
        dy = dy / scale
    return dx, dy


def apply_xy_translation(y: np.ndarray, dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    arr = np.asarray(y, dtype=np.float64)
    out = np.zeros_like(arr)
    for idx in range(arr.shape[2]):
        from skimage.transform import AffineTransform, warp

        transform = AffineTransform(translation=(dx[idx], dy[idx]))
        out[:, :, idx] = warp(
            arr[:, :, idx],
            inverse_map=transform.inverse,
            order=1,
            mode="constant",
            cval=0.0,
            clip=False,
            preserve_range=True,
        )
    return out


def simulate_ca_video(*args: Any, **kwargs: Any) -> tuple[list[str | None], str | None, list[np.ndarray], str | None, list[np.ndarray], list[Any]]:
    opt = int_var(*args, **kwargs)
    rng = ensure_rng(opt["seed"])
    outpath = get_out_path(opt)
    if outpath is not None:
        outpath.mkdir(parents=True, exist_ok=True)
    bl = create_baseline(opt, rng=rng)
    a_gt, ba_gt, c, s, lfp = create_neuron_data(opt, rng=rng)
    opt["Nneu2"] = a_gt[0].shape[2]
    c = add_remapping_drifting(c, opt, rng=rng)
    a: list[np.ndarray] = []
    ba: list[np.ndarray] = []
    mot: list[np.ndarray] = []
    bl_sessions = np.zeros((opt["d"][0] - 40, opt["d"][1] - 40, opt["ses"]), dtype=np.float64)
    first_bl, first_a, first_ba, first_d = add_nrmotion(
        bl[:, :, 0],
        a_gt[0],
        ba_gt[0],
        0,
        opt["plotme"],
        opt["translation_misalignment"],
        opt["motion_sz"],
        rng=rng,
    )
    bl_sessions[:, :, 0] = first_bl
    a.append(first_a)
    ba.append(first_ba)
    mot.append(first_d)
    for idx in range(1, opt["ses"]):
        dm, da, dba, d = add_nrmotion(
            bl[:, :, idx],
            a_gt[idx],
            ba_gt[idx],
            opt["NR_misalignment"],
            opt["plotme"],
            opt["translation_misalignment"],
            opt["motion_sz"],
            rng=rng,
        )
        bl_sessions[:, :, idx] = dm
        a.append(da)
        ba.append(dba)
        mot.append(d)
    bl_uint8 = v2uint8(bl_sessions)
    d1, d2 = bl_uint8.shape[:2]
    gt_motion_xy: list[Any] = [None for _ in range(opt["ses"])]
    file_name = datetime.utcnow().strftime("%y%m%d_%H%M%S")
    ix1 = 0
    f = opt["F"]
    ix2 = f
    pnr = opt["PNR"]
    mult = 0.8
    save_videos = bool(opt["save_videos"])
    save_mat = bool(opt["save_mat"])
    save_gt = bool(opt["save_GT"])
    pth_out: list[str | None] = [None for _ in range(opt["ses"])]
    batch_sz = opt["batch_sz"]
    gt_store: list[np.ndarray] = []
    for session in progress(range(opt["ses"]), enabled=True):
        if opt["session_motion_std"] > 0:
            seed_i = opt["seed"] + session + 1 if opt["session_motion_seed"] is None else opt["session_motion_seed"] + session
            dx, dy = make_xy_motion_ar(
                f,
                opt["session_motion_phi"],
                opt["session_motion_std"],
                opt["session_motion_max"],
                seed_i,
            )
            gt_motion_xy[session] = (dx, dy)
        else:
            dx = np.array([])
            dy = np.array([])
            gt_motion_xy[session] = []
        mc_tag = "_mc" if opt["session_motion_std"] == 0 else ""
        if save_videos:
            if opt["save_avi"]:
                pth = (outpath or Path.cwd()) / f"{file_name}_ses{session:02d}{mc_tag}.avi"
                pth_out[session] = str(pth)
                video_frames: list[np.ndarray] = []
            else:
                pth = (outpath or Path.cwd()) / f"{file_name}_ses{session:02d}{mc_tag}"
                pth_out[session] = str(pth.with_suffix(".h5"))
        t_a = crop_edges(a_gt[session])
        tb_a = crop_edges(ba_gt[session])
        session_frames = np.arange(ix1, ix2)
        gt_batches: list[np.ndarray] = []
        for start_idx in progress(range(0, f, batch_sz), enabled=True):
            batch_frames = np.arange(start_idx, min(start_idx + batch_sz, f))
            global_idx = session_frames[batch_frames]
            v_batch = flatten_frames(a[session]) @ c[:, global_idx]
            v_batch = v_batch.reshape((d1, d2, batch_frames.size), order="F")
            v_batch += (flatten_frames(ba[session]) @ c[:, global_idx]).reshape((d1, d2, batch_frames.size), order="F")
            v_batch = v2uint8(v_batch)
            n_batch = rng.rand(d1, d2, batch_frames.size).astype(np.float32)
            n_batch = gaussian_filter(n_batch, sigma=(0.5, 0.5, 0.5))
            n_batch = n_batch - np.mean(n_batch, axis=2, keepdims=True)
            n_batch = v2uint8(n_batch)
            fv_batch = (
                mult * bl_uint8[:, :, session][:, :, np.newaxis]
                + (1 - mult) * v_batch * (1 / (1 / pnr + 1))
                + (1 - mult) * n_batch * (1 - 1 / (1 / pnr + 1))
            )
            fv_batch = v2uint8(fv_batch)
            if opt["session_motion_std"] > 0:
                fv_batch = apply_xy_translation(fv_batch.astype(np.float64), dx[batch_frames], dy[batch_frames])
                fv_batch = v2uint8(fv_batch)
            if save_videos:
                if opt["save_avi"]:
                    for idx in range(fv_batch.shape[2]):
                        video_frames.append(fv_batch[:, :, idx] + 1)
                else:
                    saveash5(fv_batch + 1, str(pth))
            if save_gt:
                v_gt_batch = flatten_frames(t_a) @ c[:, global_idx]
                v_gt_batch = v_gt_batch.reshape((d1, d2, batch_frames.size), order="F")
                v_gt_batch += (flatten_frames(tb_a) @ c[:, global_idx]).reshape((d1, d2, batch_frames.size), order="F")
                v_gt_batch = v2uint8(v_gt_batch)
                gt_batches.append(v_gt_batch)
        if save_videos and opt["save_avi"]:
            save_as_avi(np.stack(video_frames, axis=2), str(pth))
        if save_gt:
            gt_store.append(np.concatenate(gt_batches, axis=2))
        ix1 += f
        ix2 += f
    if save_gt and gt_store:
        gt_full = np.concatenate(gt_store, axis=2)
        if opt["save_avi"]:
            save_as_avi(gt_full + 1, str((outpath or Path.cwd()) / f"{file_name}_GT.avi"))
        else:
            saveash5(gt_full + 1, str((outpath or Path.cwd()) / f"{file_name}_GT"))
    if save_mat:
        savemat(
            (outpath or Path.cwd()) / f"{file_name}_meta.mat",
            {
                "A_GT": np.asarray(a_gt, dtype=object),
                "A": np.asarray(a, dtype=object),
                "C": c,
                "S": s,
                "BL": bl_uint8,
                "opt": opt,
                "Mot": np.asarray(mot, dtype=object),
                "LFP": lfp,
                "GT_motion_xy": np.asarray(gt_motion_xy, dtype=object),
            },
            do_compression=False,
        )
    if not (save_videos or save_mat or save_gt):
        file_name = None
    return pth_out, str(outpath) if outpath is not None else None, mot, file_name, a, gt_motion_xy


Simulate_Ca_video = simulate_ca_video
