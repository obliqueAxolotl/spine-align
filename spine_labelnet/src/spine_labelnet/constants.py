"""Label vocabularies shared by every module.

Two vocabularies live here and must not be confused:

* VerSe vertebra labels -- integers in the CT masks (1 = C1 ... 24 = L5,
  25 = L6, 28 = T13). These are *anatomical names* and only the data
  pipeline sees them.
* Part classes -- the output of the part-label head: fixed vertebra parts
  (handoff Component 1, plus body and pedicles because CT sees the whole
  vertebra) followed by configurable non-vertebral bone classes (classes.py).

The network itself never sees an anatomical name. Its level head predicts a
*relative* index (lowest in-view vertebra = 0, increasing rostrally), and
naming is a separate, manual step downstream.
"""

from __future__ import annotations

IGNORE_INDEX = -100

# --------------------------------------------------------------------------
# Vertebra part classes (fixed)
# --------------------------------------------------------------------------
# The handoff's list was written for dorsal-only ultrasound; CT sees the whole
# vertebra, so body and pedicles are classes too. Non-vertebral bone classes
# (other_bone by default) are appended from the config -- see classes.py.
VERTEBRA_PART_CLASSES: list[str] = [
    "spinous_process",      # 0
    "lamina_left",          # 1
    "lamina_right",         # 2
    "transverse_left",      # 3
    "transverse_right",     # 4
    "facet_left",           # 5
    "facet_right",          # 6
    "pedicle_left",         # 7
    "pedicle_right",        # 8
    "vertebral_body",       # 9
]
PART = {name: i for i, name in enumerate(VERTEBRA_PART_CLASSES)}

# Cache value for a point that is bone but not vertebra; its training class
# comes from its bone_id through the config's bone_map.
NON_VERTEBRAL = -1

# --------------------------------------------------------------------------
# VerSe vertebra labels
# --------------------------------------------------------------------------
VERSE_LABELS: dict[int, str] = {
    1: "C1", 2: "C2", 3: "C3", 4: "C4", 5: "C5", 6: "C6", 7: "C7",
    8: "T1", 9: "T2", 10: "T3", 11: "T4", 12: "T5", 13: "T6", 14: "T7",
    15: "T8", 16: "T9", 17: "T10", 18: "T11", 19: "T12",
    20: "L1", 21: "L2", 22: "L3", 23: "L4", 24: "L5", 25: "L6",
    28: "T13",
}

# Cranio-caudal order key. T13 (label 28) sits between T12 and L1, so the
# raw label value cannot be used for ordering.
VERSE_ORDER: dict[int, float] = {k: float(k) for k in VERSE_LABELS}
VERSE_ORDER[28] = 19.5

THORACIC = {8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 28}
LUMBAR = {20, 21, 22, 23, 24, 25}
CERVICAL = {1, 2, 3, 4, 5, 6, 7}


def region_of(label: int) -> str:
    if label in CERVICAL:
        return "cervical"
    if label in THORACIC:
        return "thoracic"
    if label in LUMBAR:
        return "lumbar"
    return "unknown"


def caudal_to_rostral(labels) -> list[int]:
    """Sort VerSe labels from the most caudal (e.g. L5) to the most rostral."""
    return sorted(labels, key=lambda l: -VERSE_ORDER[int(l)])
