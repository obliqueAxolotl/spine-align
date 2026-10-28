"""`spine-labelnet check-data`: verify an extracted VerSe-Lite folder before preprocessing.

The package was written against the VerSe-Lite dataset card, not against the
files themselves, so this checks the assumptions it makes and says plainly
which ones the data breaks.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import nibabel as nib
import numpy as np

from ..constants import VERSE_LABELS
from .verse import _SUBJECT_RE, discover_scans, find_image, load_mask


def check_data(raw_dir: str | Path, levels: list[int], sample: int = 20,
               need_images: bool = False) -> tuple[list[str], bool]:
    """Return (report lines, ok). ok is False when something would break preprocessing."""
    lines: list[str] = []
    ok = True

    def say(status: str, msg: str):
        nonlocal ok
        if status == "FAIL":
            ok = False
        lines.append(f"[{status}] {msg}")

    try:
        recs = discover_scans(raw_dir)
    except (FileNotFoundError, ValueError) as e:
        say("FAIL", str(e))
        return lines, False

    say("OK" if len(recs) >= 300 else "WARN",
        f"{len(recs)} mask files found (VerSe-Lite card says 325). Example: {recs[0].mask_path.name}")

    named = sum(bool(_SUBJECT_RE.search(r.scan_id)) for r in recs)
    subjects = Counter(r.subject_id for r in recs)
    if named == len(recs):
        say("OK", f"subject IDs parsed from every name: {len(subjects)} subjects, "
                  f"{sum(c > 1 for c in subjects.values())} with more than one scan")
    else:
        say("WARN", f"{len(recs) - named} names have no 'sub-...' part; each such scan is "
                    f"treated as its own subject. Check that no patient appears twice.")

    if need_images or any(find_image(r) for r in recs[:5]):
        paired = sum(find_image(r) is not None for r in recs)
        say("OK" if paired == len(recs) else ("FAIL" if need_images else "WARN"),
            f"{paired}/{len(recs)} masks matched to a CT image (needed for extra-bone --run)")
    else:
        say("INFO", "no Images/ folder found (fine unless you will run TotalSegmentator)")

    rng = np.random.default_rng(0)
    picks = rng.choice(len(recs), min(sample, len(recs)), replace=False)
    unknown, axcodes, forms, dtypes, n_chain, spacings = Counter(), Counter(), Counter(), Counter(), [], []
    for i in picks:
        r = recs[int(i)]
        hdr = nib.load(str(r.mask_path)).header
        forms[(int(hdr["qform_code"]), int(hdr["sform_code"]))] += 1
        try:
            m = load_mask(r.mask_path)
        except Exception as e:
            say("FAIL", f"{r.mask_path.name}: cannot load ({type(e).__name__}: {e})")
            continue
        vals = [int(v) for v in np.unique(m.data) if v != 0]
        unknown.update(v for v in vals if v not in VERSE_LABELS)
        n_chain.append(sum(v in set(levels) for v in vals))
        axcodes["".join(m.axcodes)] += 1
        dtypes[str(m.data.dtype)] += 1
        spacings.append(m.spacing)

    say("OK" if not unknown else "WARN",
        f"label values outside the VerSe map: {dict(unknown) or 'none'} "
        f"(checked {len(picks)} masks; unknown values are ignored)")
    say("INFO", f"(qform_code, sform_code) seen: {dict(forms)}; geometry is read from the affine either way")
    say("OK" if set(axcodes) == {"RAS"} else "INFO",
        f"orientation after loading: {dict(axcodes)}; dtypes: {dict(dtypes)}")
    if spacings:
        sp = np.array(spacings)
        say("INFO", f"voxel spacing mm: min {np.round(sp.min(0), 2).tolist()}, "
                    f"max {np.round(sp.max(0), 2).tolist()}")
    if n_chain:
        few = sum(n < 3 for n in n_chain)
        say("OK" if few < len(n_chain) else "FAIL",
            f"thoracolumbar levels per scan (sample): median {int(np.median(n_chain))}, "
            f"{few}/{len(n_chain)} have fewer than 3 and will be skipped")
    return lines, ok
