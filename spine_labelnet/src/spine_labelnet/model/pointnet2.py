"""PointNet++ (single-scale grouping) in plain PyTorch -- no CUDA extensions.

The handoff names three backbones: a sparse-conv U-Net (recommended start),
Point Transformer v3 (stronger), PointNet++ (weaker but fastest to stand up).
This package starts with PointNet++ because it installs anywhere; the heads
and losses only see a (B, C, N) feature tensor, so swapping in a sparse-conv
backbone later touches this file and nothing else.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """points (B, N, C), idx (B, ...) -> (B, ..., C)."""
    B = points.shape[0]
    flat = idx.reshape(B, -1)
    out = torch.gather(points, 1, flat.unsqueeze(-1).expand(-1, -1, points.shape[-1]))
    return out.reshape(*idx.shape, points.shape[-1])


@torch.no_grad()
def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    B, N, _ = xyz.shape
    idx = torch.zeros(B, npoint, dtype=torch.long, device=xyz.device)
    dist = torch.full((B, N), float("inf"), device=xyz.device)
    far = torch.randint(0, N, (B,), device=xyz.device)
    ar = torch.arange(B, device=xyz.device)
    for i in range(npoint):
        idx[:, i] = far
        c = xyz[ar, far].unsqueeze(1)
        dist = torch.minimum(dist, ((xyz - c) ** 2).sum(-1))
        far = dist.argmax(-1)
    return idx


@torch.no_grad()
def knn_ball(xyz: torch.Tensor, centers: torch.Tensor, k: int, radius: float) -> torch.Tensor:
    """k nearest neighbours, with neighbours beyond `radius` replaced by the nearest one."""
    d = torch.cdist(centers, xyz)                      # (B, S, N)
    dist, idx = d.topk(min(k, xyz.shape[1]), dim=-1, largest=False)
    far = dist > radius
    idx = torch.where(far, idx[..., :1].expand_as(idx), idx)
    return idx


class SetAbstraction(nn.Module):
    def __init__(self, npoint: int, radius: float, nsample: int, in_ch: int, mlp: list[int],
                 sampling: str = "fps"):
        super().__init__()
        self.npoint, self.radius, self.nsample, self.sampling = npoint, radius, nsample, sampling
        layers, c = [], in_ch + 3
        for h in mlp:
            layers += [nn.Conv2d(c, h, 1, bias=False), nn.BatchNorm2d(h), nn.ReLU(inplace=True)]
            c = h
        self.mlp = nn.Sequential(*layers)
        self.out_ch = c

    def forward(self, xyz, feat):
        # xyz (B, N, 3), feat (B, N, C)
        B, N, _ = xyz.shape
        S = min(self.npoint, N)
        if self.sampling == "fps":
            cidx = farthest_point_sample(xyz, S)
        else:
            cidx = torch.stack([torch.randperm(N, device=xyz.device)[:S] for _ in range(B)])
        new_xyz = index_points(xyz, cidx)                       # (B, S, 3)
        nidx = knn_ball(xyz, new_xyz, self.nsample, self.radius)  # (B, S, K)
        g_xyz = (index_points(xyz, nidx) - new_xyz.unsqueeze(2)) / self.radius
        g = torch.cat([g_xyz, index_points(feat, nidx)], -1) if feat is not None else g_xyz
        g = self.mlp(g.permute(0, 3, 1, 2))                     # (B, C, S, K)
        return new_xyz, g.max(-1)[0].transpose(1, 2)            # (B, S, C)


class FeaturePropagation(nn.Module):
    def __init__(self, in_ch: int, mlp: list[int]):
        super().__init__()
        layers, c = [], in_ch
        for h in mlp:
            layers += [nn.Conv1d(c, h, 1, bias=False), nn.BatchNorm1d(h), nn.ReLU(inplace=True)]
            c = h
        self.mlp = nn.Sequential(*layers)
        self.out_ch = c

    def forward(self, xyz1, xyz2, feat1, feat2):
        # interpolate feat2 (at xyz2) onto xyz1, concat skip feat1
        d = torch.cdist(xyz1, xyz2)
        k = min(3, xyz2.shape[1])
        dist, idx = d.topk(k, dim=-1, largest=False)
        w = 1.0 / (dist + 1e-8)
        w = w / w.sum(-1, keepdim=True)
        interp = (index_points(feat2, idx) * w.unsqueeze(-1)).sum(2)
        x = torch.cat([feat1, interp], -1) if feat1 is not None else interp
        return self.mlp(x.transpose(1, 2)).transpose(1, 2)


class PointNet2Backbone(nn.Module):
    """Encoder-decoder producing per-point features plus a global scan feature.

    The global feature matters for the level head: the relative index counts
    vertebrae from the bottom of the scan, which no local neighbourhood can see.
    """

    def __init__(self, in_features: int, npoints, radii, nsamples, sa_mlps, fp_mlps,
                 sampling: str = "fps"):
        super().__init__()
        self.sa = nn.ModuleList()
        ch = [in_features]
        c = in_features
        for np_, r, ns, mlp in zip(npoints, radii, nsamples, sa_mlps):
            m = SetAbstraction(np_, r, ns, c, mlp, sampling)
            self.sa.append(m)
            c = m.out_ch
            ch.append(c)
        self.fp = nn.ModuleList()
        c_up = ch[-1]
        for i, mlp in enumerate(fp_mlps):
            skip = ch[-2 - i]
            m = FeaturePropagation(c_up + skip, mlp)
            self.fp.append(m)
            c_up = m.out_ch
        self.out_ch = c_up
        self.global_ch = ch[-1]

    def forward(self, xyz, feat):
        xyzs, feats = [xyz], [feat]
        for m in self.sa:
            xyz, feat = m(xyz, feat)
            xyzs.append(xyz); feats.append(feat)
        global_feat = feats[-1].max(1)[0]                       # (B, C)
        up = feats[-1]
        for i, m in enumerate(self.fp):
            lvl = len(self.sa) - 1 - i
            up = m(xyzs[lvl], xyzs[lvl + 1], feats[lvl], up)
        return up, global_feat                                  # (B, N, C), (B, Cg)
