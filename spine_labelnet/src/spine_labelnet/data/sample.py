"""CT surface sampler: surface cache -> one labelled training sample.

No ultrasound modelling. Each sample is the full bone surface extracted from
the CT segmentation over a window of contiguous vertebrae, with partial
vertebrae allowed at either end of the window, plus whatever rib, sacrum and
other bone falls in that window (when extra-bone masks were preprocessed),
cropped to a random left-right half-width. Augmentation is geometric only:

* small per-vertebra and global rotations (pose variety)
* left/right mirroring (bilateral part labels swap)

Targets (handoff, Component 1):

    part       PART class per point (body and pedicles included)
    level      relative index, lowest in-view vertebra = 0, rostral increasing
    offset     mm from the point to its vertebra's full-mask centroid
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from ..classes import PartSpace
from ..config import SampleConfig
from ..constants import IGNORE_INDEX, NON_VERTEBRAL

MAX_VERT = 16


def rodrigues(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def random_rotation(rng: np.random.Generator, max_deg: float) -> np.ndarray:
    if max_deg <= 0:
        return np.eye(3)
    return rodrigues(rng.normal(size=3), np.deg2rad(rng.uniform(-max_deg, max_deg)))


def contiguous_runs(idx: np.ndarray) -> list[np.ndarray]:
    if len(idx) == 0:
        return []
    cuts = np.where(np.diff(idx) != 1)[0] + 1
    return np.split(idx, cuts)


def sample_surface(cache: dict, cfg: SampleConfig, space: PartSpace, rng: np.random.Generator,
                   repose_fn: Callable | None = None) -> dict:
    labels = cache["labels"]
    M = len(labels)
    trunc = cache["truncated"].astype(bool)
    cents = cache["centroids"].astype(np.float64).copy()
    frames = cache["frames"].astype(np.float64).copy()
    pts = cache["points"].astype(np.float64).copy()
    nrm = cache["normals"].astype(np.float64).copy()
    part_all = cache["part"].astype(np.int64)
    # Non-vertebral bone: training class from its exact identity via the config's bone_map.
    if "bone_id" in cache:
        nv = part_all == NON_VERTEBRAL
        mapped = space.bone_lut[cache["bone_id"].astype(np.int64)[nv]]
        part_all[nv] = np.where(mapped >= 0, mapped, IGNORE_INDEX)

    lut = np.full(64, -1, np.int64)
    lut[labels.astype(np.int64)] = np.arange(M)
    vidx_all = lut[cache["vert_label"].astype(np.int64)]
    vidx_all[cache["vert_label"] == 0] = -1

    # ---- 1. window of contiguous, untruncated vertebrae (caudal -> rostral order)
    runs = contiguous_runs(np.where(~trunc)[0]) or [np.arange(M)]
    lens = np.array([len(r) for r in runs], float)
    run = runs[rng.choice(len(runs), p=lens / lens.sum())]
    k_hi = min(cfg.max_vertebrae, len(run))
    k = int(rng.integers(min(cfg.min_vertebrae, k_hi), k_hi + 1))
    s0 = int(rng.integers(0, len(run) - k + 1))
    win = run[s0:s0 + k]

    # ---- 2. pose augmentation
    if repose_fn is not None:
        pts, nrm, cents, frames = repose_fn(pts, nrm, cents, frames, vidx_all, rng)
    else:
        R = np.stack([random_rotation(rng, cfg.per_vertebra_rot_deg) for _ in range(M)])
        m = vidx_all >= 0
        vi = vidx_all[m]
        pts[m] = np.einsum("nij,nj->ni", R[vi], pts[m] - cents[vi]) + cents[vi]
        nrm[m] = np.einsum("nij,nj->ni", R[vi], nrm[m])
        frames = np.einsum("mij,mjk->mik", R, frames)
    center = cents[win].mean(0)
    Rg = random_rotation(rng, cfg.global_rot_deg)
    pts = (pts - center) @ Rg.T
    nrm = nrm @ Rg.T
    cents = (cents - center) @ Rg.T
    frames = np.einsum("ij,mjk->mik", Rg, frames)

    # ---- 3. crop along the spine; ends may cut through a vertebra
    if k > 1:
        axis = cents[win[-1]] - cents[win[0]]
        spacing = np.linalg.norm(np.diff(cents[win], axis=0), axis=1).mean()
    else:
        axis, spacing = frames[win[0]][:, 2].copy(), 30.0
    axis /= np.linalg.norm(axis)
    xdir = frames[win][:, :, 0].mean(0); xdir -= axis * (xdir @ axis); xdir /= np.linalg.norm(xdir)
    s = pts @ axis
    sc = cents @ axis
    lo = sc[win[0]] - rng.uniform(*cfg.end_extension) * spacing
    # A window that ends at the scan's lowest vertebra often continues into the
    # sacrum in practice; extend it there some of the time so the class is seen.
    if win[0] == 0 and rng.random() < cfg.sacrum_extension_prob:
        lo -= rng.uniform(*cfg.sacrum_extension) * spacing
    hi = sc[win[-1]] + rng.uniform(*cfg.end_extension) * spacing
    keep = (s >= lo) & (s <= hi)
    if cfg.lateral_extent_mm is not None:
        keep &= np.abs(pts @ xdir) <= rng.uniform(*cfg.lateral_extent_mm)
    sel = np.where(keep)[0]
    if len(sel) < 100:
        raise RuntimeError("empty crop")

    p, n = pts[sel], nrm[sel]
    vidx = vidx_all[sel]
    part = part_all[sel].copy()

    # ---- 4. relative level index: a vertebra is in view when enough of its surface is
    total = np.bincount(vidx_all[vidx_all >= 0], minlength=M)
    shown = np.bincount(vidx[vidx >= 0], minlength=M)
    frac = np.where(total > 0, shown / np.maximum(total, 1), 0.0)
    in_view = np.where(frac >= cfg.in_view_min_fraction)[0]
    level_of_v = np.full(M, np.nan)
    if len(in_view):
        lo_v, hi_v = in_view.min(), in_view.max()
        level_of_v[lo_v:hi_v + 1] = np.arange(hi_v - lo_v + 1)

    has_v = vidx >= 0
    level = np.full(len(sel), np.nan)
    level[has_v] = level_of_v[vidx[has_v]]
    offset = np.zeros((len(sel), 3))
    offset[has_v] = cents[vidx[has_v]] - p[has_v]
    offset_mask = has_v & ~trunc[np.maximum(vidx, 0)]
    vlabel = np.where(has_v, labels[np.maximum(vidx, 0)], 0).astype(np.int64)

    # ---- 5. left/right mirror augmentation
    gt_c = cents.copy()
    if rng.random() < cfg.lr_flip_prob:
        H = np.eye(3) - 2 * np.outer(xdir, xdir)
        p, n, offset, gt_c = p @ H, n @ H, offset @ H, gt_c @ H
        swap = space.lr_swap_lut()
        part = np.where(part >= 0, swap[np.maximum(part, 0)], part)

    # ---- 6. resample to a fixed size, centre
    idx = rng.choice(len(p), cfg.num_points, replace=len(p) < cfg.num_points)
    mu = p[idx].mean(0)

    gt = np.where(~np.isnan(level_of_v) & ~trunc)[0][:MAX_VERT]
    gt_centroids = np.zeros((MAX_VERT, 3), np.float32)
    gt_level = np.full(MAX_VERT, -1, np.int64)
    gt_label = np.zeros(MAX_VERT, np.int64)
    gt_valid = np.zeros(MAX_VERT, bool)
    gt_centroids[:len(gt)] = gt_c[gt] - mu
    gt_level[:len(gt)] = level_of_v[gt].astype(np.int64)
    gt_label[:len(gt)] = labels[gt]
    gt_valid[:len(gt)] = True

    lvl = level[idx]
    return {
        "xyz": (p[idx] - mu).astype(np.float32),
        "normals": n[idx].astype(np.float32),
        "part": part[idx].astype(np.int64),
        "level": np.nan_to_num(lvl, nan=0.0).astype(np.float32),
        "level_mask": ~np.isnan(lvl),
        "offset": offset[idx].astype(np.float32),
        "offset_mask": offset_mask[idx],
        "vert_label": vlabel[idx],
        "gt_centroids": gt_centroids, "gt_level": gt_level,
        "gt_label": gt_label, "gt_valid": gt_valid,
    }
