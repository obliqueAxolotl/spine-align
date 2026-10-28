"""The labelling network: backbone + four per-point heads.

    part      (B, N, C)   logits; C = vertebra parts + configured bone classes
    level_mu  (B, N)      relative level coordinate (continuous)
    level_logvar (B, N)   log-variance of that coordinate
    offset    (B, N, 3)   mm, from the point to its vertebra's centroid
    conf_logit (B, N)     logit of P(point's predicted part and level are correct)

The network never outputs a pose. Two calibration scalars live in buffers and
are fitted after training on held-out data (see calibrate.py):
`conf_temperature` divides the confidence logit, `level_std_scale` scales the
predicted level standard deviation.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..config import ModelConfig
from .pointnet2 import PointNet2Backbone


def _head(c_in: int, hidden: int, c_out: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv1d(c_in, hidden, 1, bias=False), nn.BatchNorm1d(hidden), nn.ReLU(inplace=True),
        nn.Dropout(dropout), nn.Conv1d(hidden, c_out, 1))


class SpineLabelNet(nn.Module):
    def __init__(self, cfg: ModelConfig, num_parts: int):
        super().__init__()
        self.cfg = cfg
        s = cfg.coord_scale_mm
        self.backbone = PointNet2Backbone(
            cfg.in_features, cfg.sa_npoints, [r / s for r in cfg.sa_radii_mm], cfg.sa_nsample,
            cfg.sa_mlps, cfg.fp_mlps, cfg.sampling)
        c, cg, h = self.backbone.out_ch, self.backbone.global_ch, cfg.head_hidden
        self.global_proj = nn.Sequential(nn.Linear(cg, h), nn.ReLU(inplace=True))
        c_in = c + h
        self.num_parts = num_parts
        self.part_head = _head(c_in, h, num_parts, cfg.dropout)
        self.level_head = _head(c_in, h, 2, cfg.dropout)
        self.offset_head = _head(c_in, h, 3, cfg.dropout)
        self.conf_head = _head(c_in, h, 1, cfg.dropout)
        self.register_buffer("conf_temperature", torch.ones(()))
        self.register_buffer("level_std_scale", torch.ones(()))

    def forward(self, xyz_mm: torch.Tensor, normals: torch.Tensor) -> dict[str, torch.Tensor]:
        s = self.cfg.coord_scale_mm
        xyz = xyz_mm / s
        feat = torch.cat([xyz, normals], -1) if self.cfg.in_features == 6 else normals
        point_feat, global_feat = self.backbone(xyz, feat)
        g = self.global_proj(global_feat).unsqueeze(1).expand(-1, point_feat.shape[1], -1)
        x = torch.cat([point_feat, g], -1).transpose(1, 2)        # (B, C, N)

        level = self.level_head(x)
        logvar = level[:, 1].clamp(self.cfg.logvar_min, self.cfg.logvar_max)
        return {
            "part": self.part_head(x).transpose(1, 2),
            "level_mu": level[:, 0],
            "level_logvar": logvar,
            "offset": self.offset_head(x).transpose(1, 2) * s,
            "conf_logit": self.conf_head(x)[:, 0],
        }

    @torch.no_grad()
    def calibrated(self, out: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Probabilities / std devs with the fitted calibration applied."""
        return {
            "part_prob": out["part"].float().softmax(-1),
            "level_mu": out["level_mu"].float(),
            "level_std": torch.exp(0.5 * out["level_logvar"].float()) * self.level_std_scale,
            "offset": out["offset"].float(),
            "conf": torch.sigmoid(out["conf_logit"].float() / self.conf_temperature),
        }
