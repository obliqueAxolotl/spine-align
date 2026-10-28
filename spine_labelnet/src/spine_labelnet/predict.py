"""Inference on a scan: per-point targets for the kinematic solver.

Output .npz (all in the input's coordinates, mm):

    points, normals             the (thinned) input surface
    part, part_prob             part label and its probability vector
    level_mu, level_std         continuous relative level coordinate and calibrated sigma
    level_index                 decoded discrete relative index (-1 for non-vertebra points)
    conf                        calibrated confidence -> the solver's data-term weight
    vote_centroids (K,3)        one centroid per decoded level, from the points' votes
    vote_level (K,), vote_count relative index of each, and how many points voted
    order_consistent            False flags the scan for review (level order vs position)
    free_vote_clusters          vertebrae found by clustering votes blind (diagnostic only:
                                over-splits on whole volumes, so not a review flag)

Anatomical naming is NOT produced here: it comes from the manual naming
provider (click one vertebra, name it), per the handoff.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from .classes import part_colors, part_space
from .data.io import load_points, write_ply
from .data.surface import voxel_thin
from .evaluate import cluster_votes, decode_levels, order_consistent, vertebra_centroids


@torch.no_grad()
def predict_points(model, cfg, points: np.ndarray, normals: np.ndarray, device,
                   max_points: int | None = None, seed: int = 0) -> dict:
    model.eval()
    max_points = max_points or cfg.sample.num_points * 2
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(points), min(max_points, len(points)), replace=False)
    mu = points[idx].mean(0)
    xyz = torch.tensor(points[idx] - mu, dtype=torch.float32, device=device)[None]
    nrm = torch.tensor(normals[idx], dtype=torch.float32, device=device)[None]
    cal = model.calibrated(model(xyz, nrm))
    sub = {k: v[0].cpu().numpy() for k, v in cal.items()}

    # propagate from the network's subset to every input point
    nn = cKDTree(points[idx]).query(points)[1]
    part_prob = sub["part_prob"][nn]
    part = part_prob.argmax(-1)
    level_mu, level_std = sub["level_mu"][nn], sub["level_std"][nn]
    offset, conf = sub["offset"][nn], sub["conf"][nn]

    space = part_space(cfg)
    is_vert = np.isin(part, sorted(space.vertebra_parts))
    level_index = np.where(is_vert, decode_levels(level_mu, is_vert, "peaks"), -1)
    votes = points[is_vert] + offset[is_vert]
    centers, vote_level, n_votes = vertebra_centroids(
        votes, level_index[is_vert], conf[is_vert], cfg.eval.vote_cluster_radius_mm,
        cfg.eval.min_cluster_points)
    # Diagnostic only: blind clustering of the same votes (over-splits on whole volumes).
    free, _ = cluster_votes(votes, cfg.eval.vote_cluster_radius_mm, cfg.eval.min_cluster_points)
    return {
        "points": points.astype(np.float32), "normals": normals.astype(np.float32),
        "part": part.astype(np.int16), "part_prob": part_prob.astype(np.float16),
        "level_mu": level_mu.astype(np.float32), "level_std": level_std.astype(np.float32),
        "level_index": level_index.astype(np.int16), "conf": conf.astype(np.float32),
        "vote_centroids": centers.astype(np.float32), "vote_level": vote_level,
        "vote_count": n_votes, "free_vote_clusters": np.array(len(free)),
        "order_consistent": np.array(order_consistent(centers, vote_level)),
        "part_classes": np.array(space.names),
    }


def predict_file(model, cfg, in_path: str, out_path: str, device, ply: bool = True) -> dict:
    pts, nrm = load_points(in_path, cfg.data.surface_spacing_mm, cfg.data.ct_bone_threshold_hu)
    keep = voxel_thin(pts, cfg.data.surface_spacing_mm)
    pts, nrm = pts[keep], nrm[keep]
    res = predict_points(model, cfg, pts, nrm, device)
    np.savez_compressed(out_path, **res)
    if ply:
        write_ply(Path(out_path).with_suffix(".ply"), res["points"],
                  part_colors(len(res["part_classes"]))[res["part"]],
                  {"level_mu": res["level_mu"], "conf": res["conf"]})
    return res
