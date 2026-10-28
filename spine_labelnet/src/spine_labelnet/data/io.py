"""Reading scans for inference (CT NIfTI / STL / npz) and writing PLY for inspection."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from .surface import _sample_mesh


def read_stl(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (vertices (F*3,3), faces (F,3)) from a binary or ASCII STL."""
    raw = Path(path).read_bytes()
    is_binary = len(raw) >= 84 and len(raw) == 84 + 50 * struct.unpack("<I", raw[80:84])[0]
    if is_binary:
        n = struct.unpack("<I", raw[80:84])[0]
        rec = np.frombuffer(raw[84:84 + 50 * n],
                            dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
        verts = rec["v"].reshape(-1, 3).astype(np.float64)
    else:
        vals = [list(map(float, line.split()[1:4]))
                for line in raw.decode(errors="ignore").splitlines()
                if line.strip().startswith("vertex")]
        verts = np.array(vals, np.float64)
    faces = np.arange(len(verts)).reshape(-1, 3)
    return verts, faces


def stl_to_points(path: str | Path, spacing_mm: float = 1.0, seed: int = 0):
    verts, faces = read_stl(path)
    tri = verts[faces]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    fn /= np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12
    vnormals = np.repeat(fn, 3, axis=0)        # unshared vertices: face normal per corner
    s = _sample_mesh(verts, faces, vnormals, spacing_mm, np.random.default_rng(seed))
    return s.points, s.normals


def orient_normals_outward(points: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """Flip all normals if they mostly point inward (STL winding is not guaranteed)."""
    radial = points - points.mean(0)
    return -normals if np.einsum("ij,ij->i", radial, normals).mean() < 0 else normals


def ct_to_points(path: str | Path, threshold: float, spacing_mm: float = 1.0,
                 smoothing_vox: float = 0.75, seed: int = 0):
    """Bone surface of a CT volume (values >= threshold), in RAS world mm.

    With a raw CT, `threshold` is in HU (~200 for bone). With a segmentation
    mask, pass 0.5. Thresholded CT includes ribs, pelvis and everything else
    that is bone; crop to the spine first if the field of view is large.
    """
    from scipy import ndimage
    from .surface import binary_surface
    from .verse import check_affine
    import nibabel as nib
    img = nib.as_closest_canonical(nib.load(str(path)))
    check_affine(img, path)
    vol = np.asanyarray(img.dataobj)
    bone = vol >= threshold
    if not bone.any():
        raise ValueError(f"{path}: nothing at or above threshold {threshold}")
    slc = ndimage.find_objects(bone.astype(np.uint8))[0]
    offset = np.array([s.start for s in slc], dtype=np.float64)
    s = binary_surface(bone[slc], img.affine, offset, spacing_mm, smoothing_vox,
                       np.random.default_rng(seed))
    return s.points, s.normals


def load_points(path: str | Path, spacing_mm: float = 1.0, ct_threshold: float = 200.0,
                orient: bool = True) -> tuple[np.ndarray, np.ndarray]:
    path = Path(path)
    name = path.name.lower()
    if name.endswith((".nii", ".nii.gz")):
        return ct_to_points(path, ct_threshold, spacing_mm)
    if name.endswith(".stl"):
        pts, nrm = stl_to_points(path, spacing_mm)
    elif name.endswith(".npz"):
        z = np.load(path)
        pts, nrm = z["points"].astype(np.float32), z["normals"].astype(np.float32)
    else:
        raise ValueError(f"Unsupported input: {path.name} (use .nii.gz, .stl or .npz)")
    return pts, (orient_normals_outward(pts, nrm) if orient else nrm)


def write_ply(path: str | Path, points: np.ndarray, colors: np.ndarray | None = None,
              scalars: dict[str, np.ndarray] | None = None) -> None:
    """ASCII PLY with optional colours and extra float properties (opens in Slicer/CloudCompare)."""
    scalars = scalars or {}
    n = len(points)
    header = ["ply", "format ascii 1.0", f"element vertex {n}",
              "property float x", "property float y", "property float z"]
    if colors is not None:
        header += ["property uchar red", "property uchar green", "property uchar blue"]
    header += [f"property float {k}" for k in scalars]
    header.append("end_header")
    cols = [points.astype(np.float32)]
    if colors is not None:
        cols.append(colors.astype(np.float32))
    cols += [np.asarray(v, np.float32)[:, None] for v in scalars.values()]
    data = np.concatenate(cols, axis=1)
    fmt = ["%.3f"] * 3 + (["%d"] * 3 if colors is not None else []) + ["%.4f"] * len(scalars)
    with open(path, "w") as f:
        f.write("\n".join(header) + "\n")
        np.savetxt(f, data, fmt=fmt)
