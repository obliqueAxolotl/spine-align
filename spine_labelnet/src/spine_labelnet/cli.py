"""Command line: spine-labelnet <command> ...

    download     fetch VerSe-Lite masks from the Hugging Face Hub
    check-data   verify an extracted VerSe-Lite folder against what the package expects
    extra-bone   non-vertebral bone masks from TotalSegmentator (optional, before preprocess)
    preprocess   masks -> per-scan surface caches, manifest, subject split
    inventory    scan/level counts the handoff asks for before scaling up
    preview      write training samples as coloured PLY for spot-checking parts and levels
    train        train the labelling network
    calibrate    fit confidence temperature and level-sigma scale on held-out samples
    evaluate     metrics report on the val or test split
    predict      run on CT (.nii.gz), .stl or .npz scans and write per-point solver targets
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np


def _cfg(args):
    from .config import load_config
    return load_config(args.config, args.override)


def cmd_download(args):
    from .data.verse import download
    download(args.out, keep_zips=args.keep_zips, include_images=args.images)


def cmd_check_data(args):
    from .data.check import check_data
    cfg = _cfg(args)
    lines, ok = check_data(cfg.data.raw_dir, cfg.data.levels, args.sample, args.images)
    print("\n".join(lines))
    print("\nready for preprocess" if ok else "\nNOT ready: fix the FAIL lines first")
    if not ok:
        raise SystemExit(1)


def cmd_extra_bone(args):
    from .data.extra_bone import build_extra_masks
    cfg = _cfg(args)
    if not cfg.data.extra_masks_dir:
        raise SystemExit("set data.extra_masks_dir (config or -o data.extra_masks_dir=...)")
    res = build_extra_masks(cfg.data.raw_dir, cfg.data.extra_masks_dir, ts_dir=args.ts_dir,
                            run=args.run, structures=cfg.data.extra_structures, fast=args.fast,
                            device=args.device, overwrite=args.overwrite, limit=args.limit)
    print(json.dumps({k: (v if k != "failed" else v[:20]) for k, v in res.items()}, indent=2))


def cmd_preprocess(args):
    from .data.preprocess import run_preprocess
    run_preprocess(_cfg(args), workers=args.workers, limit=args.limit, overwrite=args.overwrite)


def cmd_inventory(args):
    from .data.preprocess import inventory
    print(json.dumps(inventory(_cfg(args)), indent=2))


def cmd_preview(args):
    from .data.dataset import SpineSurfaceDataset, cache_paths_for
    from .classes import part_colors, part_space
    from .data.io import write_ply
    from .constants import IGNORE_INDEX
    cfg = _cfg(args)
    files = cache_paths_for(cfg.data.cache_dir, cfg.data.split_file, args.split, cfg.data.min_levels_per_scan)
    space = part_space(cfg)
    ds = SpineSurfaceDataset(files, cfg.sample, space, 1, fixed=True, base_seed=args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    for i in range(min(args.n, len(ds))):
        s = {k: v.numpy() for k, v in ds[i].items()}
        part = s["part"].copy()
        col = np.where((part == IGNORE_INDEX)[:, None], 200, part_colors(space.num)[np.clip(part, 0, space.num - 1)])
        lvl = np.where(s["level_mask"], s["level"], -1)
        write_ply(out / f"sample_{i:03d}.ply", s["xyz"], col, {"level": lvl})
    print(f"wrote {min(args.n, len(ds))} samples to {out}")


def cmd_train(args):
    from .train import train
    train(_cfg(args), resume=args.resume)


def _loader(cfg, split, samples, base_seed, batch_size=None):
    from torch.utils.data import DataLoader
    from .data.dataset import SpineSurfaceDataset, cache_paths_for
    files = cache_paths_for(cfg.data.cache_dir, cfg.data.split_file, split, cfg.data.min_levels_per_scan)
    from .classes import part_space
    ds = SpineSurfaceDataset(files, cfg.sample, part_space(cfg), samples, fixed=True,
                             base_seed=base_seed)
    return DataLoader(ds, batch_size or cfg.train.batch_size, num_workers=cfg.train.num_workers)


def cmd_calibrate(args):
    import torch
    from .calibrate import calibrate
    from .evaluate import run_inference
    from .train import load_checkpoint, resolve_device
    device = resolve_device(args.device)
    model, cfg, ck = load_checkpoint(args.ckpt, device)
    model.conf_temperature.fill_(1.0); model.level_std_scale.fill_(1.0)
    # Different seeds from the validation samples used for model selection.
    res = run_inference(model, _loader(cfg, "val", args.samples, 20_000), device)
    fitted = calibrate(model, res)
    out = args.out or str(Path(args.ckpt).with_name("calibrated.pt"))
    ck["model"] = model.state_dict(); ck["calibration"] = fitted
    torch.save(ck, out)
    print(json.dumps(fitted, indent=2), "\nsaved", out)


def cmd_evaluate(args):
    from .evaluate import compute_metrics, run_inference, write_report
    from .train import load_checkpoint, resolve_device
    device = resolve_device(args.device)
    model, cfg, _ = load_checkpoint(args.ckpt, device)
    seed = 30_000 if args.split == "test" else 10_000
    res = run_inference(model, _loader(cfg, args.split, args.samples, seed), device)
    metrics = compute_metrics(res, cfg)
    path = write_report(metrics, args.out or Path(args.ckpt).parent, f"eval_{args.split}")
    print(path.read_text())


def cmd_predict(args):
    from .predict import predict_file
    from .train import load_checkpoint, resolve_device
    device = resolve_device(args.device)
    model, cfg, _ = load_checkpoint(args.ckpt, device)
    if args.threshold is not None:
        cfg.data.ct_bone_threshold_hu = args.threshold
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    for p in args.inputs:
        stem = Path(p).name.split(".")[0]
        res = predict_file(model, cfg, p, str(out / f"{stem}_labels.npz"), device)
        print(f"{p}: {len(res['points'])} points, {len(res['vote_centroids'])} vertebrae "
              f"(levels {res['vote_level'].tolist()}), order consistent: {bool(res['order_consistent'])}, "
              f"blind vote clusters: {int(res['free_vote_clusters'])}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="spine-labelnet", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def with_cfg(p):
        p.add_argument("-c", "--config", default=None, help="YAML config")
        p.add_argument("-o", "--override", action="append", default=[], help="key.path=value")
        return p

    p = sub.add_parser("download"); p.add_argument("--out", default="data/verse-lite")
    p.add_argument("--keep-zips", action="store_true"); p.add_argument("--images", action="store_true")
    p.set_defaults(fn=cmd_download)

    p = with_cfg(sub.add_parser("check-data"))
    p.add_argument("--sample", type=int, default=20, help="masks to open and inspect")
    p.add_argument("--images", action="store_true", help="require a CT image for every mask")
    p.set_defaults(fn=cmd_check_data)

    p = with_cfg(sub.add_parser("extra-bone"))
    p.add_argument("--ts-dir", help="existing TotalSegmentator output: <ts-dir>/<scan_id>/<class>.nii.gz")
    p.add_argument("--run", action="store_true", help="run TotalSegmentator where output is missing")
    p.add_argument("--fast", action="store_true", help="TotalSegmentator 3 mm fast model")
    p.add_argument("--device", default="gpu"); p.add_argument("--overwrite", action="store_true")
    p.add_argument("--limit", type=int); p.set_defaults(fn=cmd_extra_bone)

    p = with_cfg(sub.add_parser("preprocess")); p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int); p.add_argument("--overwrite", action="store_true")
    p.set_defaults(fn=cmd_preprocess)

    p = with_cfg(sub.add_parser("inventory")); p.set_defaults(fn=cmd_inventory)

    p = with_cfg(sub.add_parser("preview")); p.add_argument("--split", default="train")
    p.add_argument("--n", type=int, default=5); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="preview"); p.set_defaults(fn=cmd_preview)

    p = with_cfg(sub.add_parser("train")); p.add_argument("--resume"); p.set_defaults(fn=cmd_train)

    for name, fn in [("calibrate", cmd_calibrate), ("evaluate", cmd_evaluate)]:
        p = sub.add_parser(name); p.add_argument("--ckpt", required=True)
        p.add_argument("--samples", type=int, default=5, help="fixed samples per spine")
        p.add_argument("--device", default="auto"); p.add_argument("--out")
        if name == "evaluate":
            p.add_argument("--split", default="val", choices=["val", "test"])
        p.set_defaults(fn=fn)

    p = sub.add_parser("predict"); p.add_argument("--ckpt", required=True)
    p.add_argument("inputs", nargs="+"); p.add_argument("--out-dir", default="predictions")
    p.add_argument("--threshold", type=float, default=None,
                   help="bone threshold for NIfTI input (HU for CT, 0.5 for a mask)")
    p.add_argument("--device", default="auto"); p.set_defaults(fn=cmd_predict)

    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args.fn(args)


if __name__ == "__main__":
    main()
