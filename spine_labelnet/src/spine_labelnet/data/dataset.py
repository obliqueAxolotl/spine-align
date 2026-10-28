"""PyTorch dataset: labelled CT bone-surface samples drawn on the fly."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch
from torch.utils.data import Dataset

from ..classes import PartSpace
from ..config import SampleConfig
from .preprocess import read_manifest
from .sample import sample_surface
from .verse import load_split


@lru_cache(maxsize=48)
def _load_cache(path: str) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def cache_paths_for(cache_dir: str, split_file: str, split: str, min_complete: int,
                    fraction: float = 1.0, seed: int = 0) -> list[str]:
    """Cache files whose *subject* belongs to `split`. `fraction` subsamples subjects."""
    subjects = set(load_split(split_file)[split])
    if fraction < 1.0:
        subs = sorted(subjects)
        rng = np.random.default_rng(seed)
        subjects = set(rng.choice(subs, max(1, int(round(fraction * len(subs)))), replace=False))
    rows = read_manifest(cache_dir)
    return sorted(r["cache"] for r in rows
                  if r["subject_id"] in subjects and int(r["n_complete"]) >= min_complete)


def to_tensors(sample: dict) -> dict[str, torch.Tensor]:
    return {k: torch.as_tensor(np.asarray(v)) for k, v in sample.items()}


class SpineSurfaceDataset(Dataset):
    """Each item is a window of vertebrae cut from one VerSe-Lite spine's CT surface.

    fixed=True gives a deterministic set (sample i always uses seed base_seed+i):
    use it for validation, calibration and test so numbers are comparable
    across epochs and runs.
    """

    def __init__(self, cache_files: list[str], cfg: SampleConfig, space: PartSpace,
                 samples_per_spine: int, fixed: bool = False, base_seed: int = 0, repose_fn=None):
        if not cache_files:
            raise ValueError("no cache files for this split")
        self.files = list(cache_files)
        self.cfg = cfg
        self.space = space
        self.n = len(self.files) * samples_per_spine
        self.fixed = fixed
        self.base_seed = base_seed
        self.repose_fn = repose_fn

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
        cache = _load_cache(self.files[spine])
        for attempt in range(10):
            try:
                out = sample_surface(cache, self.cfg, self.space, rng, self.repose_fn)
                break
            except RuntimeError:
                if attempt == 9:
                    raise
        out["sample_id"] = np.array(i)
        return to_tensors(out)
