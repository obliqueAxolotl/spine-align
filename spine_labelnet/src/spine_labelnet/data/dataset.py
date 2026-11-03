"""PyTorch dataset: labelled CT bone-surface samples drawn on the fly."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..classes import PartSpace
from ..config import SampleConfig
from .preprocess import read_manifest
from .sample import sample_surface
from .verse import load_split


def _read_cache(path: str) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


class _SpineCache:
    """Most-recently-used spine caches kept in memory, per process.

    Each DataLoader worker holds its own copy, so memory is about
    (num_workers + 1) x max_items x the size of one spine's cache.
    """

    def __init__(self, max_items: int):
        self.max_items = max(1, int(max_items))
        self._items: OrderedDict[str, dict] = OrderedDict()

    def get(self, path: str) -> dict:
        if path in self._items:
            self._items.move_to_end(path)
            return self._items[path]
        item = _read_cache(path)
        self._items[path] = item
        if len(self._items) > self.max_items:
            self._items.popitem(last=False)
        return item


def cache_paths_for(cache_dir: str, split_file: str, split: str, min_complete: int,
                    fraction: float = 1.0, seed: int = 0) -> list[str]:
    """Cache files whose *subject* belongs to `split`. `fraction` subsamples subjects."""
    subjects = set(load_split(split_file)[split])
    if fraction < 1.0:
        subs = sorted(subjects)
        rng = np.random.default_rng(seed)
        subjects = set(rng.choice(subs, max(1, int(round(fraction * len(subs)))), replace=False))
    rows = read_manifest(cache_dir)
    # Resolve each cache inside cache_dir by scan ID rather than trusting the path
    # stored in the manifest, so caches built on one machine work after being
    # copied to another (e.g. preprocessed locally, trained on a cluster).
    paths = sorted(str(Path(cache_dir) / f"{r['scan_id']}.npz") for r in rows
                   if r["subject_id"] in subjects and int(r["n_complete"]) >= min_complete)
    missing = [p for p in paths if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} cache files listed in the manifest are not in "
                                f"{cache_dir}, e.g. {missing[0]}. Copy the whole cache folder.")
    return paths


def to_tensors(sample: dict) -> dict[str, torch.Tensor]:
    return {k: torch.as_tensor(np.asarray(v)) for k, v in sample.items()}


class SpineSurfaceDataset(Dataset):
    """Each item is a window of vertebrae cut from one VerSe-Lite spine's CT surface.

    fixed=True gives a deterministic set (sample i always uses seed base_seed+i):
    use it for validation, calibration and test so numbers are comparable
    across epochs and runs.
    """

    def __init__(self, cache_files: list[str], cfg: SampleConfig, space: PartSpace,
                 samples_per_spine: int, fixed: bool = False, base_seed: int = 0, repose_fn=None,
                 cache_size: int = 48):
        if not cache_files:
            raise ValueError("no cache files for this split")
        self.files = list(cache_files)
        self.cfg = cfg
        self.space = space
        self.n = len(self.files) * samples_per_spine
        self.fixed = fixed
        self.base_seed = base_seed
        self.repose_fn = repose_fn
        self._cache = _SpineCache(cache_size)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        spine = i % len(self.files)
        if self.fixed:
            rng = np.random.default_rng(self.base_seed + i)
        else:
            # Drawn from torch's RNG so it advances every call, including inside
            # persistent DataLoader workers.
            rng = np.random.default_rng([self.base_seed, i, int(torch.randint(0, 2 ** 31, (1,)))])
        cache = self._cache.get(self.files[spine])
        for attempt in range(10):
            try:
                out = sample_surface(cache, self.cfg, self.space, rng, self.repose_fn)
                break
            except RuntimeError:
                if attempt == 9:
                    raise
        out["sample_id"] = np.array(i)
        return to_tensors(out)
