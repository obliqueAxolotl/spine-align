"""Non-vertebral bone from TotalSegmentator, kept with its exact identity.

VerSe segments vertebrae only, but a thresholded CT, and later an ultrasound
surface scan, also contain ribs, the sacrum, the iliac crest and more. If the
network never sees that bone labelled, it assigns it to the nearest
vertebra class.

TotalSegmentator's `total` task segments those structures by name. This
module combines its per-class output into one mask per scan whose values are
codes into BONE_STRUCTURES (code = index + 1), so every point remembers
exactly which bone it came from. How structures are grouped into training
classes is decided later, from the config (see classes.py) -- by default all
of them are `other_bone`.

Codes are stable: new structures are only ever appended to BONE_STRUCTURES,
so existing masks and caches stay valid.

Structures are matched by *name* (TotalSegmentator's per-class output, one
`<class>.nii.gz` per structure), so the TotalSegmentator version's integer
IDs do not matter. Check TotalSegmentator's own licence terms for the task
you use before building anything commercial on its output.
"""

from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path

import nibabel as nib
import numpy as np

from .verse import NIFTI_SUFFIXES, _strip_nifti, check_affine, discover_scans, find_image

log = logging.getLogger(__name__)

# Append only -- the position is the code stored in masks and caches.
BONE_STRUCTURES: list[str] = (
    ["sacrum"]
    + [f"rib_left_{n}" for n in range(1, 13)]
    + [f"rib_right_{n}" for n in range(1, 13)]
    + ["hip_left", "hip_right", "sternum",
       "scapula_left", "scapula_right", "clavicula_left", "clavicula_right",
       "femur_left", "femur_right", "humerus_left", "humerus_right",
       "costal_cartilages", "skull", "vertebrae_S1"]
)
BONE_CODE = {name: i + 1 for i, name in enumerate(BONE_STRUCTURES)}

# Extracted unless data.extra_structures says otherwise. Costal cartilage is
# not bone (and rarely above a bone threshold); skull is outside the chain.
DEFAULT_STRUCTURES = [s for s in BONE_STRUCTURES if s not in ("costal_cartilages", "skull")]


def bone_name(code: int) -> str:
    return BONE_STRUCTURES[code - 1] if code > 0 else "vertebra"


def combine_folder(folder: str | Path, structures: list[str] | None = None):
    """Per-class TotalSegmentator NIfTIs -> (uint8 code volume, affine)."""
    folder = Path(folder)
    wanted = set(structures if structures is not None else DEFAULT_STRUCTURES)
    unknown = wanted - set(BONE_CODE)
    if unknown:
        raise ValueError(f"not in BONE_STRUCTURES (append them there first): {sorted(unknown)}")
    vol, ref = None, None
    for p in sorted(folder.iterdir()):
        if not p.name.endswith(NIFTI_SUFFIXES):
            continue
        name = _strip_nifti(p.name)
        if name not in wanted:
            continue
        img = nib.load(str(p))
        if ref is None:
            ref = img
            check_affine(img, p)
            vol = np.zeros(img.shape, np.uint8)
        elif img.shape != ref.shape or not np.allclose(img.affine, ref.affine, atol=1e-3):
            raise ValueError(f"{p}: grid differs from the other classes in {folder}")
        m = np.asanyarray(img.dataobj) > 0
        vol[m & (vol == 0)] = BONE_CODE[name]
    if ref is None:
        raise FileNotFoundError(f"no TotalSegmentator bone classes in {folder}")
    return vol, ref.affine


def save_combined(vol: np.ndarray, affine: np.ndarray, out_path: str | Path) -> Path:
    img = nib.Nifti1Image(vol, affine)
    img.set_qform(affine, code=1)
    img.set_sform(affine, code=1)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(img, str(out_path))
    return out_path


def run_totalsegmentator(image: Path, out_dir: Path, structures: list[str],
                         fast: bool = False, device: str = "gpu") -> Path:
    """Run TotalSegmentator into out_dir. Output is written to a temporary
    folder and renamed when complete, so a job killed half-way (time limit,
    preemption) never leaves a partial folder that a rerun would trust."""
    try:
        from totalsegmentator.python_api import totalsegmentator
    except ImportError as e:
        raise ImportError("TotalSegmentator is not installed: pip install TotalSegmentator") from e
    tmp = out_dir.with_name(out_dir.name + ".partial")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    totalsegmentator(str(image), str(tmp), roi_subset=list(structures), fast=fast, device=device,
                     quiet=True)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    tmp.rename(out_dir)
    return out_dir


def build_extra_masks(raw_dir: str, extra_masks_dir: str, ts_dir: str | None = None,
                      run: bool = False, structures: list[str] | None = None,
                      fast: bool = False, device: str = "gpu", overwrite: bool = False,
                      limit: int | None = None, shard: tuple[int, int] = (0, 1)) -> dict:
    """Write <extra_masks_dir>/<scan_id>.nii.gz (BONE_STRUCTURES codes) for every scan.

    ts_dir: TotalSegmentator output laid out as <ts_dir>/<scan_id>/<class>.nii.gz.
    run=True runs TotalSegmentator on Images/<scan_id>.nii.gz where that output is missing.
    shard=(i, n) processes every n-th scan starting at i (after `limit`), so n
    processes -- one per GPU -- split the scans between them without overlap.
    """
    i_shard, n_shard = shard
    if not (n_shard >= 1 and 0 <= i_shard < n_shard):
        raise ValueError(f"shard must be (i, n) with 0 <= i < n, got {shard}")
    structures = structures if structures is not None else DEFAULT_STRUCTURES
    out_root = Path(extra_masks_dir)
    ts_root = Path(ts_dir) if ts_dir else out_root / "_totalsegmentator"
    done, skipped, failed = 0, 0, []
    records = discover_scans(raw_dir)
    records = records[:limit] if limit else records
    records = records[i_shard::n_shard]
    tag = f"[shard {i_shard + 1}/{n_shard}] " if n_shard > 1 else ""
    log.info("%s%d scans to check", tag, len(records))
    t_start = time.time()
    for i, rec in enumerate(records):
        out = out_root / f"{rec.scan_id}.nii.gz"
        if out.exists() and not overwrite:
            skipped += 1
            continue
        folder = ts_root / rec.scan_id
        try:
            if not folder.exists():
                if not run:
                    raise FileNotFoundError(f"no TotalSegmentator output at {folder} (pass --run)")
                image = find_image(rec)
                if image is None:
                    raise FileNotFoundError("CT image not found; download with --images")
                run_totalsegmentator(image, folder, structures, fast, device)
            vol, affine = combine_folder(folder, structures)
            save_combined(vol, affine, out)
            done += 1
            found = sorted({bone_name(int(c)) for c in np.unique(vol) if c > 0})
            rate = (time.time() - t_start) / done
            log.info("%s[%d/%d] %s: %d structures (%s); %.0f s/scan, ~%.1f h left", tag, i + 1, len(records),
                     rec.scan_id, len(found), ", ".join(found[:6]) + (" ..." if len(found) > 6 else ""),
                     rate, rate * (len(records) - i - 1) / 3600)
        except Exception as e:
            failed.append({"scan_id": rec.scan_id, "error": f"{type(e).__name__}: {e}"})
            log.warning("%s: %s", rec.scan_id, e)
    return {"written": done, "skipped_existing": skipped, "failed": failed}
