"""Training regime.

* Samples are cut on the fly from the training subjects' CT bone surfaces
  (random vertebra windows, small rotations, left/right mirroring), so every
  epoch sees new crops of the same anatomy.
* Validation uses a fixed-seed set of samples of the validation subjects.
* The confidence head switches on after `loss.conf_warmup_epochs`, because its
  target depends on the other heads' predictions being meaningful.
* AdamW, linear warmup then cosine decay, gradient clipping, AMP on CUDA.
* Best checkpoint is chosen on part mIoU + per-vertebra level-index accuracy.
"""

from __future__ import annotations

import json
import logging
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .classes import part_space
from .config import Config, save_config
from .data.dataset import SpineSurfaceDataset, cache_paths_for
from .evaluate import compute_metrics, run_inference, score, write_report
from .losses import compute_losses, estimate_class_weights
from .model.labelnet import SpineLabelNet

log = logging.getLogger(__name__)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def seed_everything(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def build_datasets(cfg: Config):
    d, t = cfg.data, cfg.train
    train_files = cache_paths_for(d.cache_dir, d.split_file, "train", d.min_levels_per_scan,
                                  t.train_fraction, t.seed)
    val_files = cache_paths_for(d.cache_dir, d.split_file, "val", d.min_levels_per_scan)
    space = part_space(cfg)
    train_ds = SpineSurfaceDataset(train_files, cfg.sample, space, t.samples_per_spine, base_seed=t.seed)
    val_ds = SpineSurfaceDataset(val_files, cfg.sample, space, t.val_samples_per_spine, fixed=True,
                                 base_seed=10_000)
    log.info("train spines %d (%d samples/epoch), val spines %d (%d fixed samples)",
             len(train_files), len(train_ds), len(val_files), len(val_ds))
    return train_ds, val_ds


def build_model(cfg: Config) -> SpineLabelNet:
    return SpineLabelNet(cfg.model, part_space(cfg).num)


def save_checkpoint(path: Path, model, cfg: Config, epoch: int, metrics: dict | None,
                    optimizer=None, class_weights=None):
    torch.save({"model": model.state_dict(), "config": cfg.to_dict(), "epoch": epoch,
                "metrics": metrics,
                "optimizer": optimizer.state_dict() if optimizer is not None else None,
                "class_weights": class_weights}, path)


def load_checkpoint(path: str | Path, device="cpu"):
    from .config import config_from_dict
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = config_from_dict(ck["config"])
    model = SpineLabelNet(cfg.model, part_space(cfg).num)
    model.load_state_dict(ck["model"])
    return model.to(device), cfg, ck


def lr_lambda(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return (step + 1) / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, p)))


def train(cfg: Config, resume: str | None = None) -> Path:
    t = cfg.train
    out_dir = Path(t.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out_dir / "config.yaml")
    seed_everything(t.seed)
    device = resolve_device(t.device)
    log.info("device: %s", device)

    train_ds, val_ds = build_datasets(cfg)
    loader = DataLoader(train_ds, t.batch_size, shuffle=True, num_workers=t.num_workers,
                        drop_last=True, persistent_workers=t.num_workers > 0)
    val_loader = DataLoader(val_ds, t.batch_size, shuffle=False, num_workers=t.num_workers)

    model = build_model(cfg).to(device)
    log.info("parameters: %.2fM", sum(p.numel() for p in model.parameters()) / 1e6)

    if cfg.loss.class_weights == "auto":
        cw = estimate_class_weights(train_ds, part_space(cfg).num)
    elif cfg.loss.class_weights == "none":
        cw = None
    else:
        cw = torch.tensor(cfg.loss.class_weights, dtype=torch.float32)
    if cw is not None:
        log.info("class weights: %s", [round(x, 2) for x in cw.tolist()])
        cw = cw.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=t.lr, weight_decay=t.weight_decay)
    steps_per_epoch = max(1, len(loader))
    total = t.epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: lr_lambda(s, total, t.warmup_epochs * steps_per_epoch))
    use_amp = t.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch, best = 0, -float("inf")
    if resume:
        ck = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        if ck.get("optimizer"):
            opt.load_state_dict(ck["optimizer"])
        start_epoch = ck["epoch"] + 1
        for _ in range(start_epoch * steps_per_epoch):
            sched.step()
        best = score(ck["metrics"]) if ck.get("metrics") else best

    log_path = out_dir / "metrics.jsonl"
    for epoch in range(start_epoch, t.epochs):
        model.train()
        conf_active = epoch >= cfg.loss.conf_warmup_epochs
        sums: dict[str, float] = {}
        t0 = time.time()
        for step, batch in enumerate(loader):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                out = model(batch["xyz"], batch["normals"])
            losses = compute_losses(out, batch, cfg.loss, cw, conf_active)
            opt.zero_grad(set_to_none=True)
            scaler.scale(losses["loss"]).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), t.grad_clip)
            scaler.step(opt)
            scaler.update()
            sched.step()
            for k, v in losses.items():
                sums[k] = sums.get(k, 0.0) + float(v.detach())
            if (step + 1) % t.log_every == 0:
                log.info("epoch %d step %d/%d loss %.4f (part %.3f level %.3f offset %.3f conf %.3f) lr %.2e",
                         epoch, step + 1, steps_per_epoch, losses["loss"].item(), losses["part"].item(),
                         losses["level"].item(), losses["offset"].item(), losses["conf"].item(),
                         sched.get_last_lr()[0])
        train_stats = {k: v / steps_per_epoch for k, v in sums.items()}

        results = run_inference(model, val_loader, device)
        metrics = compute_metrics(results, cfg)
        sc = score(metrics)
        rec = {"epoch": epoch, "time_s": round(time.time() - t0, 1), "train": train_stats,
               "val_score": sc, "val_part_mIoU": metrics["part"]["mIoU"],
               "val_part_acc": metrics["part"]["accuracy"],
               "val_level_index_acc": metrics.get("level", {}).get("vertebra_index_acc_peaks"),
               "val_level_mae": metrics.get("level", {}).get("point_mae"),
               "val_offset_median_mm": metrics.get("offset", {}).get("median_mm"),
               "val_centroid_median_mm": metrics.get("centroid_votes", {}).get("median_err_mm")}
        log.info("epoch %d: %s", epoch, json.dumps({k: v for k, v in rec.items() if k != "train"}))
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

        cw_list = cw.tolist() if cw is not None else None
        save_checkpoint(out_dir / "last.pt", model, cfg, epoch, metrics, opt, cw_list)
        if sc > best:
            best = sc
            save_checkpoint(out_dir / "best.pt", model, cfg, epoch, metrics, None, cw_list)
            write_report(metrics, out_dir, "val_best")
            log.info("new best (score %.4f)", sc)
    return out_dir / "best.pt"
