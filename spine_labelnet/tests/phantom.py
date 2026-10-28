"""A crude synthetic lumbar spine in VerSe format, for tests and smoke runs.

Not anatomy: just enough structure (body, pedicles, arch, spinous and
transverse processes, facets) that every code path has something to chew on.
Written with qform_code=1, sform_code=0 like VerSe, on an anisotropic grid.
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np


def _vertebra(u, v, w):
    body = (u / 20) ** 2 + ((v - 14) / 15) ** 2 <= 1
    body &= np.abs(w) <= 12
    ped = (np.abs(u) >= 8) & (np.abs(u) <= 14) & (v >= -6) & (v <= 2) & (np.abs(w) <= 6)
    r = np.sqrt(u ** 2 + (v + 6) ** 2)
    arch = (r >= 9) & (r <= 15) & (v <= -6) & (np.abs(w) <= 9)
    sp_top = 6 - 0.0 * v
    sp_bot = -10 + 0.35 * (v + 18)
    spin = (np.abs(u) <= 3.5) & (v >= -40) & (v <= -14) & (w <= sp_top) & (w >= sp_bot)
    tp = (np.abs(u) >= 13) & (np.abs(u) <= 36) & (v >= -10) & (v <= -3) & (np.abs(w) <= 5)
    sup = (np.abs(u) >= 10) & (np.abs(u) <= 16) & (v >= -16) & (v <= -8) & (w >= 6) & (w <= 14)
    inf = (np.abs(u) >= 8) & (np.abs(u) <= 14) & (v >= -19) & (v <= -11) & (w >= -15) & (w <= -7)
    return body | ped | arch | spin | tp | sup | inf


def make_phantom(path: str | Path, labels=(19, 20, 21, 22, 23, 24), seed: int = 0,
                 spacing=(0.9, 0.9, 1.5), truncate_top: bool = True,
                 bottom_margin: float = 25.0, half_width: float = 55.0) -> Path:
    rng = np.random.default_rng(seed)
    n = len(labels)
    step = 36.0
    zs = np.arange(n)[::-1] * step                      # labels caudal-last -> z increasing rostrally
    order = np.argsort(zs)
    extent_z = (zs.min() - bottom_margin, zs.max() + (10 if truncate_top else 25))
    xs = np.arange(-half_width, half_width, spacing[0]); ys = np.arange(-50, 45, spacing[1])
    zz = np.arange(*extent_z, spacing[2])
    X, Y, Z = np.meshgrid(xs, ys, zz, indexing="ij")
    vol = np.zeros(X.shape, np.uint16)
    curve = rng.uniform(4, 10)
    for i in order:
        lab, z0 = labels[i], zs[i]
        x0 = curve * np.sin(z0 / 80.0)
        rot = np.deg2rad(rng.uniform(-6, 6))            # axial rotation
        tilt = np.deg2rad(rng.uniform(-4, 4))           # lateral bend
        c, s = np.cos(rot), np.sin(rot)
        dx, dy, dz = X - x0, Y, Z - z0
        u = c * dx + s * dy
        v = -s * dx + c * dy
        ct, st = np.cos(tilt), np.sin(tilt)
        u, w = ct * u + st * dz, -st * u + ct * dz
        m = _vertebra(u, v, w) & (vol == 0)
        vol[m] = lab
    aff = np.diag([*spacing, 1.0])
    aff[:3, 3] = [xs[0], ys[0], zz[0]]
    img = nib.Nifti1Image(vol, aff)
    img.set_qform(aff, code=1)
    img.set_sform(np.zeros((4, 4)), code=0)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(img, str(path))
    return path


def write_phantom_stl(mask_path: str | Path, stl_path: str | Path) -> Path:
    """Whole-spine surface as binary STL (stands in for a SpineUs export)."""
    import struct
    from skimage import measure
    img = nib.load(str(mask_path))
    vol = (np.asanyarray(img.dataobj) > 0).astype(np.float32)
    verts, faces, _, _ = measure.marching_cubes(np.pad(vol, 1), 0.5)
    verts = (verts - 1) @ img.affine[:3, :3].T + img.affine[:3, 3]
    tri = verts[faces]
    with open(stl_path, "wb") as f:
        f.write(b"\0" * 80 + struct.pack("<I", len(tri)))
        for t in tri:
            nrm = np.cross(t[1] - t[0], t[2] - t[0])
            f.write(struct.pack("<12fH", *nrm, *t.reshape(-1), 0))
    return Path(stl_path)


def write_phantom_totalsegmentator(mask_path: str | Path, out_dir: str | Path) -> Path:
    """Fake TotalSegmentator per-class output for a phantom: sacrum, hips, numbered ribs.

    Ribs leave each thoracic vertebra's transverse process and curve forward;
    the sacrum and hips sit below the lowest vertebra (needs bottom_margin >= 45).
    Files are written on the mask's grid, as TotalSegmentator does.
    """
    img = nib.load(str(mask_path))
    vol = np.asanyarray(img.dataobj)
    A = img.affine
    I, J, K = np.meshgrid(*[np.arange(n) for n in vol.shape], indexing="ij")
    X = A[0, 0] * I + A[0, 3]; Y = A[1, 1] * J + A[1, 3]; Z = A[2, 2] * K + A[2, 3]
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    def save(name, m):
        m = m & (vol == 0)
        nib.save(nib.Nifti1Image(m.astype(np.uint8), A), str(out_dir / f"{name}.nii.gz"))

    cz = {int(l): Z[vol == l].mean() for l in np.unique(vol) if l > 0}
    cx = {l: X[vol == l].mean() for l in cz}
    for lab, z0 in cz.items():
        if 8 <= lab <= 19:                                  # T1..T12 carry ribs 1..12
            n = lab - 7
            for side, sgn in (("left", -1), ("right", 1)):
                u = sgn * (X - cx[lab])
                y_c = -6 + 0.02 * (u - 36) ** 2             # curves anteriorly
                rib = (u >= 36) & (np.abs(Y - y_c) <= 3) & (np.abs(Z - z0 + 0.1 * (u - 36)) <= 3)
                save(f"rib_{side}_{n}", rib)
    low = min(cz, key=lambda l: cz[l])
    z_low = cz[low] - 20
    sac = (np.abs(X - cx[low]) <= 22) & (Y >= -18) & (Y <= 15) & (Z <= z_low) & (Z >= z_low - 25)
    save("sacrum", sac)
    for side, sgn in (("left", -1), ("right", 1)):
        u = sgn * (X - cx[low])
        save(f"hip_{side}", (u >= 28) & (u <= 50) & (Y >= -25) & (Y <= 10)
             & (Z <= z_low + 10) & (Z >= z_low - 30))
    return out_dir
