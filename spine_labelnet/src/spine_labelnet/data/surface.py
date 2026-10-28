"""CT segmentation -> per-vertebra surface point clouds, centroids and frames.

Each vertebra mask is smoothed slightly, meshed with marching cubes in voxel
space, mapped to world millimetres through the full affine (so oblique and
anisotropic volumes are handled), and resampled to an even point spacing.

The centroid is the centroid of the *full* vertebra mask, body included --
the target the centroid-offset head votes toward (handoff, Component 1).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from skimage import measure

from ..constants import VERSE_ORDER, caudal_to_rostral
from .verse import MaskVolume

log = logging.getLogger(__name__)


@dataclass
class Surface:
    points: np.ndarray      # (N, 3) world mm
    normals: np.ndarray     # (N, 3) outward unit normals


def _sample_mesh(verts: np.ndarray, faces: np.ndarray, vnormals: np.ndarray,
                 spacing: float, rng: np.random.Generator) -> Surface:
    """Area-uniform sampling of a triangle mesh, then grid thinning to ~`spacing`."""
    tri = verts[faces]                                       # (F, 3, 3)
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    n = int(np.ceil(area.sum() / (spacing ** 2) * 2.0))      # oversample, thin below
    if n == 0:
        return Surface(np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32))
    fidx = rng.choice(len(faces), size=n, p=area / area.sum())
    r1, r2 = rng.random(n), rng.random(n)
    s = np.sqrt(r1)
    b = np.stack([1 - s, s * (1 - r2), s * r2], axis=1)       # barycentric
    pts = np.einsum("nk,nkd->nd", b, tri[fidx])
    nrm = np.einsum("nk,nkd->nd", b, vnormals[faces[fidx]])
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9
    keep = voxel_thin(pts, spacing)
    return Surface(pts[keep].astype(np.float32), nrm[keep].astype(np.float32))


def voxel_thin(points: np.ndarray, cell: float) -> np.ndarray:
    """Indices keeping one point per `cell`-sized grid cell."""
    keys = np.floor(points / cell).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return np.sort(idx)


def binary_surface(binary: np.ndarray, affine: np.ndarray, offset_vox: np.ndarray,
                   spacing_mm: float, smoothing_vox: float,
                   rng: np.random.Generator) -> Surface:
    """Mesh a binary crop and return world-space surface samples with outward normals."""
    vol = ndimage.gaussian_filter(binary.astype(np.float32), smoothing_vox) if smoothing_vox > 0 \
        else binary.astype(np.float32)
    vol = np.pad(vol, 1)
    if vol.max() < 0.5:
        return Surface(np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32))
    verts, faces, _, _ = measure.marching_cubes(vol, level=0.5)
    verts = verts - 1.0                                      # undo pad
    # Outward normal = -grad (values fall from 1 inside to 0 outside).
    grads = np.stack(np.gradient(vol), axis=0)
    g = np.stack([ndimage.map_coordinates(grads[d], (verts + 1.0).T, order=1) for d in range(3)], 1)
    A = affine[:3, :3]
    world = (verts + offset_vox) @ A.T + affine[:3, 3]
    n_world = -g @ np.linalg.inv(A)                           # covariant transform of a gradient
    n_world /= np.linalg.norm(n_world, axis=1, keepdims=True) + 1e-9
    return _sample_mesh(world, faces, n_world, spacing_mm, rng)


def touches_border(slc: tuple[slice, ...], shape: tuple[int, ...]) -> bool:
    return any(s.start == 0 or s.stop == dim for s, dim in zip(slc, shape))


@dataclass
class VertebraSurfaces:
    labels: np.ndarray          # (M,) VerSe labels, caudal -> rostral
    centroids: np.ndarray       # (M, 3) world mm, full-mask centroid
    truncated: np.ndarray       # (M,) bool, mask touches the field-of-view edge
    surfaces: dict[int, Surface]


def extract_vertebrae(mask: MaskVolume, levels: list[int], spacing_mm: float,
                      smoothing_vox: float, seed: int = 0) -> VertebraSurfaces:
    rng = np.random.default_rng(seed)
    data = mask.data
    present = [int(l) for l in np.unique(data) if int(l) in set(levels)]
    objects = ndimage.find_objects(data.astype(np.int32))
    labels, cents, trunc, surfs = [], [], [], {}
    for lab in caudal_to_rostral(present):
        slc = objects[lab - 1]
        if slc is None:
            continue
        pad = 3
        slc_p = tuple(slice(max(s.start - pad, 0), min(s.stop + pad, dim))
                      for s, dim in zip(slc, data.shape))
        crop = data[slc_p] == lab
        if crop.sum() < 50:
            log.warning("label %d has only %d voxels; skipped", lab, crop.sum())
            continue
        offset = np.array([s.start for s in slc_p], dtype=np.float64)
        com = np.array(ndimage.center_of_mass(crop)) + offset
        centroid = mask.affine[:3, :3] @ com + mask.affine[:3, 3]
        surf = binary_surface(crop, mask.affine, offset, spacing_mm, smoothing_vox, rng)
        if len(surf.points) == 0:
            continue
        labels.append(lab)
        cents.append(centroid)
        trunc.append(touches_border(slc, data.shape))
        surfs[lab] = surf
    return VertebraSurfaces(np.array(labels, np.int16),
                            np.array(cents, np.float32).reshape(-1, 3),
                            np.array(trunc, bool), surfs)


def spine_tangents(centroids: np.ndarray) -> np.ndarray:
    """Unit tangents along the centroid chain (ordered caudal -> rostral)."""
    m = len(centroids)
    if m < 2:
        return np.tile(np.array([0.0, 0.0, 1.0]), (m, 1))
    t = np.zeros_like(centroids, dtype=np.float64)
    t[1:-1] = centroids[2:] - centroids[:-2]
    t[0] = centroids[1] - centroids[0]
    t[-1] = centroids[-1] - centroids[-2]
    t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-9
    return t


def order_key(label: int) -> float:
    return VERSE_ORDER[int(label)]
