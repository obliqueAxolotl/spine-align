"""Rule-based part parcellation in each vertebra's own frame.

VerSe-Lite has one mask per vertebra, not per part, so the part labels the
network learns from are produced here. The handoff offers two routes:
register a part-labelled template per vertebra type, or apply rules in the
vertebra's own frame. This module is the rule route.

!!! These rules are a first draft of the labelling guide, not the guide. !!!
Write the guide once, tune `data.parcel` to match it, and spot-check the
output (`spine-labelnet preview`) before trusting it.

Frame convention (RAS world, patient supine in CT):
    x = patient right, y = anterior, z = rostral (along the local spine tangent).
The frame's axial rotation is refined by maximising left-right mirror symmetry.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np
from scipy.spatial import cKDTree

from ..constants import PART


@dataclass
class ParcelParams:
    midline_halfwidth_mm: float = 3.0     # strip used to find the canal along the midline
    min_canal_gap_mm: float = 5.0
    spinous_halfwidth_mm: float = 5.0     # |x| below this, behind the canal -> spinous process
    tp_lateral_fraction: float = 0.62     # |x| above this fraction of the lateral extent -> transverse process
    body_margin_mm: float = 2.0           # anterior to canal by more than this -> body
    facet_height_fraction: float = 0.55   # |z| beyond this fraction of posterior half-height -> facet
    facet_min_lateral_mm: float = 6.0
    symmetry_search_deg: float = 30.0
    symmetry_step_deg: float = 2.0

    @classmethod
    def from_dict(cls, d: dict | None) -> "ParcelParams":
        p = cls()
        for f in fields(cls):
            if d and f.name in d:
                setattr(p, f.name, type(getattr(p, f.name))(d[f.name]))
        return p


def initial_frame(tangent: np.ndarray) -> np.ndarray:
    """3x3 matrix with columns (x=right, y=anterior, z=rostral)."""
    z = tangent / np.linalg.norm(tangent)
    a = np.array([0.0, 1.0, 0.0])
    y = a - z * (a @ z)
    if np.linalg.norm(y) < 1e-3:                       # tangent along A/P: fall back to S
        y = np.array([0.0, 0.0, 1.0]) - z * z[2]
    y /= np.linalg.norm(y)
    x = np.cross(y, z)
    return np.stack([x, y, z], axis=1)


def refine_axial_rotation(local: np.ndarray, params: ParcelParams,
                          rng: np.random.Generator) -> float:
    """Angle (rad) about z that best maps the point set onto its own mirror image."""
    pts = local if len(local) <= 2000 else local[rng.choice(len(local), 2000, replace=False)]
    tree_src = pts
    best, best_cost = 0.0, np.inf
    for deg in np.arange(-params.symmetry_search_deg, params.symmetry_search_deg + 1e-6,
                         params.symmetry_step_deg):
        t = np.deg2rad(deg)
        c, s = np.cos(t), np.sin(t)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        p = tree_src @ R                               # coordinates in rotated frame
        mirrored = p * np.array([-1.0, 1.0, 1.0])
        cost = cKDTree(p).query(mirrored)[0].mean()
        if cost < best_cost:
            best, best_cost = t, cost
    return best


def vertebra_frame(points: np.ndarray, centroid: np.ndarray, tangent: np.ndarray,
                   params: ParcelParams, rng: np.random.Generator) -> np.ndarray:
    F = initial_frame(tangent)
    local = (points - centroid) @ F
    t = refine_axial_rotation(local, params, rng)
    c, s = np.cos(t), np.sin(t)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return F @ R


def _canal_extent(v_mid: np.ndarray, min_gap: float) -> tuple[float, float] | None:
    """Largest empty run along the AP axis among midline points -> (posterior, anterior) edges."""
    if len(v_mid) < 10:
        return None
    v = np.sort(v_mid)
    gaps = np.diff(v)
    i = int(np.argmax(gaps))
    if gaps[i] < min_gap:
        return None
    return float(v[i]), float(v[i + 1])


def parcellate(points: np.ndarray, centroid: np.ndarray, frame: np.ndarray,
               params: ParcelParams) -> np.ndarray:
    """Return a PART id for every surface point of one vertebra."""
    loc = (points - centroid) @ frame
    u, v, w = loc[:, 0], loc[:, 1], loc[:, 2]
    au = np.abs(u)

    mid = au < params.midline_halfwidth_mm
    canal = _canal_extent(v[mid], params.min_canal_gap_mm)
    if canal is None:                                   # fallback: canal just behind the centroid
        v_post, v_ant = -18.0, -4.0
    else:
        v_post, v_ant = canal

    posterior = v <= v_ant + params.body_margin_mm
    U = np.percentile(au[posterior], 98) if posterior.any() else au.max()
    tp_x = params.tp_lateral_fraction * U

    post_core = posterior & (au < tp_x)
    if post_core.any():
        w_mid = np.median(w[post_core])
        half_h = 0.5 * (np.percentile(w[post_core], 97) - np.percentile(w[post_core], 3))
    else:
        w_mid, half_h = 0.0, 15.0

    part = np.full(len(points), PART["lamina_left"], dtype=np.int16)
    right = u > 0

    is_body = v > v_ant + params.body_margin_mm
    is_tp = ~is_body & (au >= tp_x)
    is_sp = ~is_body & ~is_tp & (au <= params.spinous_halfwidth_mm) & (v < v_post)
    is_facet = (~is_body & ~is_tp & ~is_sp & (au >= params.facet_min_lateral_mm)
                & (np.abs(w - w_mid) > params.facet_height_fraction * half_h))
    # Lateral to the canal, between its posterior and anterior edges: pedicle.
    is_ped = ~is_body & ~is_tp & ~is_sp & ~is_facet & (v >= v_post)
    is_lam = ~is_body & ~is_tp & ~is_sp & ~is_facet & ~is_ped

    part[is_lam & right] = PART["lamina_right"]
    part[is_lam & ~right] = PART["lamina_left"]
    part[is_tp & right] = PART["transverse_right"]
    part[is_tp & ~right] = PART["transverse_left"]
    part[is_facet & right] = PART["facet_right"]
    part[is_facet & ~right] = PART["facet_left"]
    part[is_sp] = PART["spinous_process"]
    part[is_ped & right] = PART["pedicle_right"]
    part[is_ped & ~right] = PART["pedicle_left"]
    part[is_body] = PART["vertebral_body"]
    return part
