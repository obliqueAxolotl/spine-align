"""The part-label class space: fixed vertebra parts + configurable non-vertebral bone.

Vertebra parts are fixed: the parcellation produces them and the solver
consumes them. Everything else -- ribs, sacrum, hips, sternum, ... -- is kept
in the cache with its exact TotalSegmentator identity (`bone_id`), and is
grouped into training classes only when samples are drawn, by the
`classes` section of the config:

    classes:
      bone: [other_bone]          # non-vertebral classes the network predicts
      bone_map:                   # first match wins; fnmatch on structure names
        - ["*", other_bone]

To split classes out later, change the config and retrain; nothing is
re-preprocessed. For example:

    classes:
      bone: [rib, sacrum, iliac, other_bone]
      bone_map:
        - ["rib_*", rib]
        - ["sacrum", sacrum]
        - ["hip_*", iliac]
        - ["*", other_bone]

Classes named `*_left` / `*_right` are paired automatically for the
left/right mirror augmentation.
"""

from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatchcase

import numpy as np

from .constants import VERTEBRA_PART_CLASSES
from .data.extra_bone import BONE_STRUCTURES


@dataclass(frozen=True)
class PartSpace:
    names: tuple[str, ...]
    bone_lut: np.ndarray            # bone_id (0..len(BONE_STRUCTURES)) -> class index, -1 unmapped

    @property
    def num(self) -> int:
        return len(self.names)

    @property
    def vertebra_parts(self) -> frozenset[int]:
        return frozenset(range(len(VERTEBRA_PART_CLASSES)))

    def index(self, name: str) -> int:
        return self.names.index(name)

    @property
    def lr_swap(self) -> dict[int, int]:
        swap = {}
        for i, n in enumerate(self.names):
            if n.endswith("_left") and n[:-5] + "_right" in self.names:
                j = self.names.index(n[:-5] + "_right")
                swap[i], swap[j] = j, i
        return swap

    def lr_swap_lut(self) -> np.ndarray:
        lut = np.arange(self.num)
        for a, b in self.lr_swap.items():
            lut[a] = b
        return lut


def build_part_space(bone_classes: list[str], bone_map: list[list[str]]) -> PartSpace:
    bone_classes = list(bone_classes)
    clash = set(bone_classes) & set(VERTEBRA_PART_CLASSES)
    if clash:
        raise ValueError(f"bone classes reuse vertebra part names: {sorted(clash)}")
    if len(set(bone_classes)) != len(bone_classes):
        raise ValueError("duplicate bone class names")
    names = tuple(VERTEBRA_PART_CLASSES) + tuple(bone_classes)
    lut = np.full(len(BONE_STRUCTURES) + 1, -1, np.int64)
    for code, struct in enumerate(BONE_STRUCTURES, start=1):
        for pattern, cls in bone_map:
            if fnmatchcase(struct, pattern):
                if cls not in bone_classes:
                    raise ValueError(f"bone_map target '{cls}' is not in classes.bone")
                lut[code] = names.index(cls)
                break
    return PartSpace(names, lut)


def part_space(cfg) -> PartSpace:
    return build_part_space(cfg.classes.bone, cfg.classes.bone_map)


_PALETTE = np.array([
    [230, 25, 75], [60, 180, 75], [0, 130, 200], [245, 130, 48], [145, 30, 180],
    [70, 240, 240], [240, 50, 230], [210, 245, 60], [0, 128, 128], [220, 190, 255],
    [128, 128, 128], [170, 110, 40], [255, 250, 200], [128, 0, 0], [170, 255, 195],
    [128, 128, 0], [255, 215, 180], [0, 0, 128]], np.uint8)


def part_colors(n: int) -> np.ndarray:
    return _PALETTE[np.arange(n) % len(_PALETTE)]
