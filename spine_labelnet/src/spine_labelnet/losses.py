"""Losses for the four heads.

    part    class-weighted cross-entropy, ignore-index for unlabelled points
    level   Gaussian negative log-likelihood on (mu, logvar)
    offset  smooth L1 in mm, on points of untruncated vertebrae
    conf    BCE against an event that can be checked: the predicted part is
            right AND the predicted level is within 0.5 of the truth
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .config import LossConfig
from .constants import IGNORE_INDEX


def confidence_target(out: dict, batch: dict) -> torch.Tensor:
    """1 where the point's predicted labels are right."""
    with torch.no_grad():
        pred_part = out["part"].argmax(-1)
        gt_part = batch["part"]
        part_ok = (gt_part == IGNORE_INDEX) | (pred_part == gt_part)
        lvl_ok = ~batch["level_mask"] | ((out["level_mu"] - batch["level"]).abs() < 0.5)
        return (part_ok & lvl_ok).float()


def level_nll(mu, logvar, target, mask):
    nll = 0.5 * (logvar + (target - mu) ** 2 * torch.exp(-logvar))
    return (nll * mask).sum() / mask.sum().clamp(min=1)


def compute_losses(out: dict, batch: dict, cfg: LossConfig,
                   class_weights: torch.Tensor | None, conf_active: bool) -> dict[str, torch.Tensor]:
    logits = out["part"].float()
    l_part = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), batch["part"].reshape(-1),
                             weight=class_weights, ignore_index=IGNORE_INDEX)

    lm = batch["level_mask"].float()
    l_level = level_nll(out["level_mu"].float(), out["level_logvar"].float(), batch["level"], lm)

    om = batch["offset_mask"].float()
    sl1 = F.smooth_l1_loss(out["offset"].float(), batch["offset"], beta=cfg.offset_beta_mm,
                           reduction="none").sum(-1)
    l_offset = (sl1 * om).sum() / om.sum().clamp(min=1)

    if conf_active:
        tgt = confidence_target(out, batch)
        l_conf = F.binary_cross_entropy_with_logits(out["conf_logit"].float(), tgt)
    else:
        l_conf = torch.zeros((), device=logits.device)

    total = (cfg.w_part * l_part + cfg.w_level * l_level
             + cfg.w_offset * l_offset + cfg.w_conf * l_conf)
    return {"loss": total, "part": l_part.detach(), "level": l_level.detach(),
            "offset": l_offset.detach(), "conf": l_conf.detach()}


def estimate_class_weights(dataset, num_parts: int, n_samples: int = 64, seed: int = 0) -> torch.Tensor:
    """Inverse-sqrt-frequency weights from a few samples; mean weight 1 over present classes."""
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(dataset), generator=g)[:n_samples]
    counts = torch.zeros(num_parts)
    for i in idx.tolist():
        p = dataset[i]["part"]
        p = p[p != IGNORE_INDEX]
        counts += torch.bincount(p, minlength=num_parts).float()
    freq = counts / counts.sum().clamp(min=1)
    w = torch.where(freq > 0, 1.0 / freq.clamp(min=1e-6).sqrt(), torch.zeros_like(freq))
    present = freq > 0
    w[present] = w[present] / w[present].mean()
    w[~present] = 1.0
    return w
