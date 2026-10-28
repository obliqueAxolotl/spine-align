"""Build per-scan surface caches from VerSe-Lite masks, plus a manifest and split.

One uncompressed .npz per scan holding everything the surface sampler needs:

    points (N,3) f32, normals (N,3) f32       world mm, RAS
    vert_label (N,) i16                        VerSe label, 0 for non-vertebral bone
    part (N,) i16                              vertebra PART id, NON_VERTEBRAL (-1) for other bone
    bone_id (N,) i16                           BONE_STRUCTURES code (extra_bone.py), 0 for vertebrae
    labels (M,) i16                            vertebrae, caudal -> rostral
    centroids (M,3) f32, frames (M,3,3) f32    full-mask centroid, local frame (cols x,y,z)
    truncated (M,) bool                        touches the CT field of view
"""

from __future__ import annotations

import csv
import json
import logging
import traceback
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from ..config import Config
from ..constants import LUMBAR, NON_VERTEBRAL, VERSE_LABELS, VERSE_ORDER
from .parcellation import ParcelParams, parcellate, vertebra_frame
from scipy import ndimage

from .surface import binary_surface, extract_vertebrae, spine_tangents
from .verse import MaskVolume, ScanRecord, discover_scans, load_mask, make_split, save_split

log = logging.getLogger(__name__)


def extra_bone_surfaces(em: MaskVolume, vertebra_mask: MaskVolume, spacing_mm: float,
                        smoothing_vox: float, rng: np.random.Generator):
    """One surface per extra-bone structure (each rib separately), VerSe taking priority."""
    data = em.data.astype(np.int32)
    same_grid = (data.shape == vertebra_mask.data.shape
                 and np.allclose(em.affine, vertebra_mask.affine, atol=1e-3))
    if same_grid:
        data = np.where(vertebra_mask.data > 0, 0, data)   # a voxel VerSe calls vertebra stays vertebra
    else:
        log.warning("extra-bone mask is on a different grid; overlap with vertebrae not removed")
    objects = ndimage.find_objects(data)
    for code, slc in enumerate(objects, start=1):
        if slc is None:
            continue
        crop = data[slc] == code
        if crop.sum() < 50:
            continue
        offset = np.array([s.start for s in slc], dtype=np.float64)
        surf = binary_surface(crop, em.affine, offset, spacing_mm, smoothing_vox, rng)
        if len(surf.points):
            yield code, surf


def process_scan(rec: ScanRecord, cfg: Config, out_dir: Path) -> dict:
    out_path = out_dir / f"{rec.scan_id}.npz"
    mask = load_mask(rec.mask_path)
    vs = extract_vertebrae(mask, cfg.data.levels, cfg.data.surface_spacing_mm,
                           cfg.data.mask_smoothing_vox, seed=0)
    params = ParcelParams.from_dict(getattr(cfg.data, "parcel", None))
    rng = np.random.default_rng(0)
    tangents = spine_tangents(vs.centroids.astype(np.float64))

    P, N, VL, PT, BI, frames = [], [], [], [], [], []
    for i, lab in enumerate(vs.labels):
        s = vs.surfaces[int(lab)]
        F = vertebra_frame(s.points, vs.centroids[i], tangents[i], params, rng)
        part = parcellate(s.points, vs.centroids[i], F, params)
        P.append(s.points); N.append(s.normals)
        VL.append(np.full(len(s.points), lab, np.int16)); PT.append(part)
        BI.append(np.zeros(len(s.points), np.int16))
        frames.append(F)

    if cfg.data.extra_masks_dir:
        extra = Path(cfg.data.extra_masks_dir) / f"{rec.scan_id}.nii.gz"
        if extra.exists():
            em = load_mask(extra)
            for code, s in extra_bone_surfaces(em, mask, cfg.data.surface_spacing_mm,
                                               cfg.data.mask_smoothing_vox, rng):
                P.append(s.points); N.append(s.normals)
                VL.append(np.zeros(len(s.points), np.int16))
                PT.append(np.full(len(s.points), NON_VERTEBRAL, np.int16))
                BI.append(np.full(len(s.points), code, np.int16))
        else:
            log.warning("%s: no extra-bone mask at %s", rec.scan_id, extra)

    if not P:
        raise ValueError("no in-chain vertebrae")
    np.savez(out_path,
             points=np.concatenate(P).astype(np.float32),
             normals=np.concatenate(N).astype(np.float32),
             vert_label=np.concatenate(VL), part=np.concatenate(PT), bone_id=np.concatenate(BI),
             labels=vs.labels, centroids=vs.centroids,
             frames=np.array(frames, np.float32).reshape(-1, 3, 3),
             truncated=vs.truncated, scan_id=rec.scan_id, subject_id=rec.subject_id)
    labs = [int(l) for l in vs.labels]
    return {
        "scan_id": rec.scan_id, "subject_id": rec.subject_id, "cache": str(out_path),
        "n_levels": len(labs), "n_complete": int((~vs.truncated).sum()),
        "levels": " ".join(VERSE_LABELS[l] for l in labs),
        "has_lumbar": any(l in LUMBAR for l in labs),
        "has_lower_thoracic": any(l in (17, 18, 19, 28) for l in labs),
        "spacing_mm": " ".join(f"{s:.2f}" for s in mask.spacing),
        "n_points": int(sum(len(p) for p in P)),
        "extra_bone_points": int(sum(len(b) for b in BI if b.any())),
    }


def _worker(args):
    rec, cfg, out_dir = args
    try:
        return process_scan(rec, cfg, out_dir)
    except Exception as e:  # keep going; report at the end
        return {"scan_id": rec.scan_id, "error": f"{type(e).__name__}: {e}",
                "trace": traceback.format_exc()}


def run_preprocess(cfg: Config, workers: int = 4, limit: int | None = None,
                   overwrite: bool = False) -> Path:
    out_dir = Path(cfg.data.cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = discover_scans(cfg.data.raw_dir)
    if limit:
        records = records[:limit]
    todo = [r for r in records if overwrite or not (out_dir / f"{r.scan_id}.npz").exists()]
    log.info("%d scans found, %d to process", len(records), len(todo))
    jobs = [(r, cfg, out_dir) for r in todo]
    results = []
    if workers > 1 and len(jobs) > 1:
        with Pool(workers) as pool:
            for i, res in enumerate(pool.imap_unordered(_worker, jobs)):
                results.append(res)
                log.info("[%d/%d] %s %s", i + 1, len(jobs), res["scan_id"], res.get("error", "ok"))
    else:
        for i, job in enumerate(jobs):
            res = _worker(job)
            results.append(res)
            log.info("[%d/%d] %s %s", i + 1, len(jobs), res["scan_id"], res.get("error", "ok"))

    # Merge with any previous manifest so incremental runs keep earlier rows.
    manifest_path = out_dir / "manifest.csv"
    rows = {}
    if manifest_path.exists():
        with open(manifest_path) as f:
            rows = {r["scan_id"]: r for r in csv.DictReader(f)}
    errors = []
    for r in results:
        if "error" in r:
            errors.append(r)
        else:
            rows[r["scan_id"]] = r
    if rows:
        keys = list(next(iter(rows.values())).keys())
        with open(manifest_path, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=keys)
            wr.writeheader()
            for r in rows.values():
                wr.writerow(r)
    if errors:
        with open(out_dir / "errors.json", "w") as f:
            json.dump(errors, f, indent=2)
        log.warning("%d scans failed; see %s", len(errors), out_dir / "errors.json")

    usable = [r for r in rows.values() if int(r["n_complete"]) >= cfg.data.min_levels_per_scan]
    split = make_split([r["subject_id"] for r in usable], cfg.data.val_fraction,
                       cfg.data.test_fraction, cfg.data.split_seed)
    save_split(split, cfg.data.split_file)
    log.info("split by subject: %s", {k: len(v) for k, v in split.items()})
    return manifest_path


def read_manifest(cache_dir: str | Path) -> list[dict]:
    with open(Path(cache_dir) / "manifest.csv") as f:
        return list(csv.DictReader(f))


def inventory(cfg: Config) -> dict:
    """The counts the handoff asks for before committing to a synthetic-data scale."""
    rows = read_manifest(cfg.data.cache_dir)
    t = lambda x: str(x).lower() == "true"
    usable = [r for r in rows if int(r["n_complete"]) >= cfg.data.min_levels_per_scan]
    level_counts: dict[str, int] = {}
    for r in rows:
        for name in r["levels"].split():
            level_counts[name] = level_counts.get(name, 0) + 1
    return {
        "scans": len(rows),
        "subjects": len({r["subject_id"] for r in rows}),
        "usable_scans": len(usable),
        "with_lumbar": sum(t(r["has_lumbar"]) for r in rows),
        "with_lower_thoracic": sum(t(r["has_lower_thoracic"]) for r in rows),
        "with_lumbar_and_lower_thoracic": sum(t(r["has_lumbar"]) and t(r["has_lower_thoracic"])
                                              for r in rows),
        "scans_per_level": dict(sorted(level_counts.items(), key=lambda kv: VERSE_ORDER[
            next(k for k, v in VERSE_LABELS.items() if v == kv[0])])),
    }
