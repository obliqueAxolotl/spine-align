"""VerSe-Lite access: download, discovery, safe NIfTI loading, subject split.

Notes carried over from the handoff and the dataset card:

* VerSe-Lite ships as five ``data-part00N.zip`` archives (~49 GB total) holding
  ``Images/`` and ``Masks/``. Only the masks are needed here -- the surface is
  extracted from segmentation, not from CT intensities -- so ``download``
  extracts ``Masks/`` members only.
* VerSe files carry ``sform_code=0, qform_code=1``. Geometry must come from
  ``img.affine`` (nibabel resolves it from the qform). Reading the sform
  directly gives a zero matrix. ``load_mask`` checks this explicitly.
* VerSe holds several scans per subject for some patients. Splits are made by
  subject ID, never by scan or by rendered sample.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np

log = logging.getLogger(__name__)

HF_REPO = "YongchengYAO/VerSe-Lite"
NIFTI_SUFFIXES = (".nii.gz", ".nii")
_SUBJECT_RE = re.compile(r"(sub-[A-Za-z]*\d+)")


# --------------------------------------------------------------------------
# Download
# --------------------------------------------------------------------------
def _folder_kind(member: str) -> str | None:
    """'mask' / 'image' from the archive path's directory names (case-insensitive)."""
    dirs = [d.lower() for d in Path(member).parts[:-1]]
    if "masks" in dirs:
        return "mask"
    if "images" in dirs:
        return "image"
    return None


def extract_zip(path: str | Path, raw_dir: str | Path, include_images: bool) -> dict:
    """Extract Masks/ (and optionally Images/) members; report what was found."""
    counts = {"mask": 0, "image": 0, "other": 0}
    examples = []
    with zipfile.ZipFile(path) as zf:
        for member in zf.namelist():
            name = Path(member).name
            if member.endswith("/") or "__MACOSX" in member or name.startswith("._"):
                continue
            kind = _folder_kind(member)
            if kind is None or not name.endswith(NIFTI_SUFFIXES):
                counts["other"] += 1
                if len(examples) < 10:
                    examples.append(member)
                continue
            counts[kind] += 1
            if kind == "mask" or include_images:
                zf.extract(member, raw_dir)
    if counts["mask"] == 0:
        raise RuntimeError(f"{path}: no NIfTI files under a Masks/ folder. Unexpected layout; "
                           f"first entries: {examples}")
    if counts["other"]:
        log.warning("%s: %d entries outside Images/ and Masks/ were skipped, e.g. %s",
                    path, counts["other"], examples[:5])
    return counts


def download(raw_dir: str | Path, keep_zips: bool = False, include_images: bool = False) -> Path:
    """Fetch VerSe-Lite from the Hugging Face Hub and extract the masks (~49 GB of zips).

    The zips are downloaded one at a time and deleted after extraction unless
    keep_zips, so peak disk use is one zip (~11 GB) plus what is extracted.
    """
    from huggingface_hub import list_repo_files, hf_hub_download

    raw_dir = Path(raw_dir)
    zip_dir = raw_dir / "_zips"
    zip_dir.mkdir(parents=True, exist_ok=True)
    parts = sorted(f for f in list_repo_files(HF_REPO, repo_type="dataset") if f.endswith(".zip"))
    log.info("VerSe-Lite archives: %s", parts)
    for part in parts:
        local = Path(hf_hub_download(HF_REPO, part, repo_type="dataset", local_dir=zip_dir))
        counts = extract_zip(local, raw_dir, include_images)
        log.info("%s: %d masks, %d images in archive", part, counts["mask"], counts["image"])
        if not keep_zips:
            local.unlink()
    return raw_dir


# --------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ScanRecord:
    scan_id: str
    subject_id: str
    mask_path: Path


def _strip_nifti(name: str) -> str:
    for suf in NIFTI_SUFFIXES:
        if name.endswith(suf):
            return name[: -len(suf)]
    return name


def subject_of(scan_id: str) -> str:
    """Subject ID from a VerSe-style name (sub-verse123...).

    If the name has no sub-... part, each scan is treated as its own subject.
    VerSe-Lite already dropped the duplicate split series, so that is close to
    true; guessing from a name prefix instead could put every scan into one
    "subject" and collapse the split.
    """
    m = _SUBJECT_RE.search(scan_id)
    return m.group(1) if m else scan_id


def discover_scans(raw_dir: str | Path) -> list[ScanRecord]:
    """Find every mask volume under ``raw_dir`` (searches for a Masks/ folder)."""
    raw_dir = Path(raw_dir)
    mask_dirs = [p for p in raw_dir.rglob("*") if p.is_dir() and p.name.lower() == "masks"] or [raw_dir]
    records = []
    for mdir in mask_dirs:
        for p in sorted(mdir.rglob("*")):
            if p.is_file() and p.name.endswith(NIFTI_SUFFIXES) and not p.name.startswith("._"):
                sid = _strip_nifti(p.name)
                records.append(ScanRecord(sid, subject_of(sid), p))
    if not records:
        raise FileNotFoundError(f"No NIfTI masks found under {raw_dir}")
    ids = [r.scan_id for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate scan IDs found; check the extracted folder layout.")
    return records


# Mask/image pairing. VerSe's own naming differs between the two
# (sub-verse004_ct vs sub-verse004_seg-vert_msk); VerSe-Lite may or may not keep it.
_MASK_SUFFIXES = ("_seg-vert_msk", "_seg-vert", "_msk", "_mask", "_seg", "_label", "_labels")
_IMAGE_SUFFIXES = ("_ct", "_img", "_image", "_0000")


def _base_name(stem: str, suffixes) -> str:
    for suf in suffixes:
        if stem.endswith(suf):
            return stem[: -len(suf)]
    return stem


def find_image(rec: "ScanRecord") -> Path | None:
    """The CT image for a mask: same name in a sibling Images/ folder, else same base name."""
    images_dir = next((d for d in rec.mask_path.parent.parent.iterdir()
                       if d.is_dir() and d.name.lower() == "images"), None)
    if images_dir is None:
        return None
    for suf in NIFTI_SUFFIXES:
        p = images_dir / f"{rec.scan_id}{suf}"
        if p.exists():
            return p
    base = _base_name(rec.scan_id, _MASK_SUFFIXES)
    for p in images_dir.iterdir():
        if p.name.endswith(NIFTI_SUFFIXES) and \
                _base_name(_strip_nifti(p.name), _IMAGE_SUFFIXES) == base:
            return p
    return None


# --------------------------------------------------------------------------
# Loading with affine checks
# --------------------------------------------------------------------------
@dataclass
class MaskVolume:
    data: np.ndarray            # (I, J, K) integer labels
    affine: np.ndarray          # 4x4 voxel -> RAS world (mm)
    spacing: np.ndarray         # (3,) mm
    axcodes: tuple[str, str, str]


def check_affine(img: nib.Nifti1Image, path: Path | str = "") -> np.ndarray:
    """Return the voxel->world affine after the sanity checks the handoff asks for."""
    hdr = img.header
    sform_code = int(hdr["sform_code"])
    qform_code = int(hdr["qform_code"])
    affine = np.asarray(img.affine, dtype=np.float64)
    if sform_code == 0 and qform_code == 0:
        raise ValueError(f"{path}: neither sform nor qform is set; geometry is undefined.")
    if sform_code == 0:
        # Expected for VerSe. Reading get_sform() here would return garbage/zeros.
        raw_sform = hdr.get_sform()
        if not np.allclose(raw_sform[:3, :3], 0):
            log.debug("%s: sform_code=0 but sform matrix non-zero; ignoring it.", path)
    det = np.linalg.det(affine[:3, :3])
    if not np.isfinite(det) or abs(det) < 1e-6:
        raise ValueError(f"{path}: degenerate affine (det={det}).")
    return affine


def load_mask(path: str | Path) -> MaskVolume:
    img = nib.load(str(path))
    check_affine(img, path)
    codes = nib.aff2axcodes(img.affine)
    if codes != ("R", "A", "S"):
        log.info("%s: orientation %s, reorienting to RAS+", path, codes)
        img = nib.as_closest_canonical(img)
    affine = check_affine(img, path)
    data = np.asanyarray(img.dataobj)
    if not np.issubdtype(data.dtype, np.integer):
        data = np.rint(data).astype(np.int32)
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    return MaskVolume(data=data, affine=affine, spacing=spacing,
                      axcodes=tuple(nib.aff2axcodes(affine)))


# --------------------------------------------------------------------------
# Subject-level split
# --------------------------------------------------------------------------
def make_split(subject_ids: list[str], val_fraction: float, test_fraction: float,
               seed: int = 0) -> dict[str, list[str]]:
    """Deterministic hash-based split of *subjects*; stable when scans are added."""
    def u(sid: str) -> float:
        h = hashlib.sha256(f"{seed}:{sid}".encode()).hexdigest()
        return int(h[:12], 16) / float(16 ** 12)

    split: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for sid in sorted(set(subject_ids)):
        x = u(sid)
        if x < test_fraction:
            split["test"].append(sid)
        elif x < test_fraction + val_fraction:
            split["val"].append(sid)
        else:
            split["train"].append(sid)
    return split


def save_split(split: dict[str, list[str]], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(split, f, indent=2)


def load_split(path: str | Path) -> dict[str, list[str]]:
    with open(path) as f:
        return json.load(f)
