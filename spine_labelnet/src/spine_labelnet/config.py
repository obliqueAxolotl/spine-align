"""Configuration: nested dataclasses loaded from YAML.

Every physical threshold lives here, in millimetres or degrees, rather than
in code. The handoff asks that constants tuned against the placeholder atlas
carry their provenance so they are
revisited instead of inherited -- see the `# provenance:` comments in
configs/default.yaml.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    raw_dir: str = "data/verse-lite"          # extracted VerSe-Lite (Masks/ inside)
    cache_dir: str = "data/cache"             # per-scan surface caches (.npz)
    split_file: str = "data/splits.json"      # subject-level split, written by `preprocess`
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    split_seed: int = 0
    # Levels the chain covers. Thoracolumbar now, cervical later: extending
    # coverage is a change to this list, not to the network.
    levels: list[int] = field(default_factory=lambda: list(range(8, 26)) + [28])
    min_levels_per_scan: int = 3              # scans with fewer in-chain levels are dropped
    surface_spacing_mm: float = 1.0           # point spacing on the extracted CT surface
    mask_smoothing_vox: float = 0.75          # Gaussian sigma before marching cubes
    extra_masks_dir: str | None = None        # combined rib/sacrum/other-bone masks (spine-labelnet extra-bone)
    extra_structures: list[str] | None = None  # TotalSegmentator structures to extract; None = default list
    ct_bone_threshold_hu: float = 200.0       # bone threshold when predicting from a raw CT image
    parcel: dict = field(default_factory=dict)  # overrides for data.parcellation.ParcelParams


@dataclass
class SampleConfig:
    num_points: int = 16384                   # points per training sample
    min_vertebrae: int = 3                    # window length range, in vertebrae
    max_vertebrae: int = 7
    end_extension: tuple[float, float] = (0.3, 1.0)  # crop past the end centroids, x vertebra spacing
    lateral_extent_mm: tuple[float, float] | None = (80.0, 150.0)  # random L-R half-width; None = no crop
    sacrum_extension_prob: float = 0.6        # window at the lowest vertebra: extend below it this often
    sacrum_extension: tuple[float, float] = (0.5, 1.5)  # by this many vertebra spacings
    in_view_min_fraction: float = 0.5         # fraction of a vertebra's surface needed to count as "in view"
    per_vertebra_rot_deg: float = 3.0         # pose augmentation (placeholder for atlas-rig re-posing)
    global_rot_deg: float = 10.0
    lr_flip_prob: float = 0.5


@dataclass
class ModelConfig:
    in_features: int = 6                      # xyz + normals
    sa_npoints: list[int] = field(default_factory=lambda: [2048, 512, 128, 32])
    sa_radii_mm: list[float] = field(default_factory=lambda: [8.0, 16.0, 32.0, 64.0])
    sa_nsample: list[int] = field(default_factory=lambda: [32, 32, 32, 16])
    sa_mlps: list[list[int]] = field(default_factory=lambda: [
        [32, 32, 64], [64, 64, 128], [128, 128, 256], [256, 256, 512]])
    fp_mlps: list[list[int]] = field(default_factory=lambda: [
        [256, 256], [256, 256], [256, 128], [128, 128, 128]])
    head_hidden: int = 128
    dropout: float = 0.3
    coord_scale_mm: float = 100.0             # xyz are divided by this before entering the net
    sampling: str = "fps"                     # "fps" or "random"
    logvar_min: float = -6.0
    logvar_max: float = 4.0


@dataclass
class LossConfig:
    w_part: float = 1.0
    w_level: float = 1.0
    w_offset: float = 0.05                    # offset loss is in mm; this keeps it on scale
    w_conf: float = 0.5
    offset_beta_mm: float = 2.0
    class_weights: str | list[float] = "auto"  # "auto" (inverse sqrt freq), "none", or a list
    conf_warmup_epochs: int = 2                # confidence target depends on predictions; let them settle


@dataclass
class TrainConfig:
    out_dir: str = "runs/default"
    epochs: int = 60
    batch_size: int = 8
    samples_per_spine: int = 20               # per epoch
    val_samples_per_spine: int = 5            # fixed-seed held-out samples
    train_fraction: float = 1.0               # for the learning-curve experiment (10/25/50/100 %)
    lr: float = 1e-3
    weight_decay: float = 1e-4
    warmup_epochs: int = 2
    grad_clip: float = 5.0
    amp: bool = True
    num_workers: int = 4
    seed: int = 0
    device: str = "auto"
    log_every: int = 20


@dataclass
class EvalConfig:
    level_cluster_gap: float = 0.5            # 1D gap that separates level clusters
    vote_cluster_radius_mm: float = 12.0
    min_cluster_points: int = 30
    conf_bins: int = 15


@dataclass
class ClassesConfig:
    # Non-vertebral classes the network predicts, after the fixed vertebra parts.
    bone: list[str] = field(default_factory=lambda: ["other_bone"])
    # [pattern, class] pairs, first match wins, fnmatch on TotalSegmentator names.
    bone_map: list[list[str]] = field(default_factory=lambda: [["*", "other_bone"]])


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    classes: ClassesConfig = field(default_factory=ClassesConfig)
    sample: SampleConfig = field(default_factory=SampleConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _merge(dc, values: dict[str, Any]):
    for key, val in values.items():
        if not hasattr(dc, key):
            raise KeyError(f"Unknown config key: {type(dc).__name__}.{key}")
        cur = getattr(dc, key)
        if dataclasses.is_dataclass(cur) and isinstance(val, dict):
            _merge(cur, val)
        else:
            if isinstance(cur, tuple) and isinstance(val, list):
                val = tuple(val)
            setattr(dc, key, val)
    return dc


def _parse_override(s: str) -> tuple[list[str], Any]:
    key, _, raw = s.partition("=")
    return key.strip().split("."), yaml.safe_load(raw)


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
    """Load defaults, then a YAML file, then `a.b=value` overrides."""
    cfg = Config()
    if path is not None:
        with open(path) as f:
            _merge(cfg, yaml.safe_load(f) or {})
    for ov in overrides or []:
        keys, val = _parse_override(ov)
        nested: dict[str, Any] = {keys[-1]: val}
        for k in reversed(keys[:-1]):
            nested = {k: nested}
        _merge(cfg, nested)
    return cfg


def config_from_dict(d: dict[str, Any]) -> Config:
    return _merge(Config(), d)


def save_config(cfg: Config, path: str | Path) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(cfg.to_dict(), f, sort_keys=False)
