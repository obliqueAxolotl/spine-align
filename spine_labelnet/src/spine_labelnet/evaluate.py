"""Validation of the labelling network in isolation (handoff, delivery phase 2).

"Validate the network in isolation before connecting it ... once the stages
are chained, a labelling bug and a solver bug look identical."

Reports, per level and never only pooled:

* part labels: accuracy, per-class IoU, mIoU, confusion matrix
* level coordinate: point MAE, z-score coverage of the predicted sigma, and the
  acceptance statistic -- relative level index correct per vertebra (target >99 %)
  -- decoded both by rounding and by 1D peak-finding
* centroid votes: per-point offset error, clustered-centroid error, missed /
  spurious clusters, and whether cluster order along the spine agrees with the
  decoded level order (the cross-check the handoff prescribes)
* confidence: ECE and Brier score, raw and after temperature scaling
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks
from scipy.spatial import KDTree

from .config import Config
from .classes import part_space
from .constants import IGNORE_INDEX, VERSE_LABELS, VERSE_ORDER
from .losses import confidence_target



# --------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------
@torch.no_grad()
def run_inference(model, loader, device, max_batches: int | None = None):
    model.eval()
    results = []
    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch["xyz"], batch["normals"])
        cal = model.calibrated(out)
        tgt = confidence_target(out, batch)
        B = batch["xyz"].shape[0]
        for b in range(B):
            r = {k: v[b].cpu().numpy() for k, v in batch.items()}
            r["part_prob"] = cal["part_prob"][b].cpu().numpy()
            r["part_pred"] = r["part_prob"].argmax(-1)
            r["level_mu"] = cal["level_mu"][b].cpu().numpy()
            r["level_std"] = cal["level_std"][b].cpu().numpy()
            r["offset_pred"] = cal["offset"][b].cpu().numpy()
            r["conf_logit"] = out["conf_logit"][b].float().cpu().numpy()
            r["conf"] = cal["conf"][b].cpu().numpy()
            r["conf_target"] = tgt[b].cpu().numpy()
            results.append(r)
    return results


# --------------------------------------------------------------------------
# Decoding
# --------------------------------------------------------------------------
def decode_levels(mu: np.ndarray, use: np.ndarray, method: str = "peaks",
                  bin_w: float = 0.05, min_sep: float = 0.6) -> np.ndarray:
    """Continuous level coordinate -> discrete relative index for every point."""
    if method == "round" or use.sum() < 10:
        return np.rint(mu).astype(np.int64)
    v = mu[use]
    edges = np.arange(v.min() - 1, v.max() + 1 + bin_w, bin_w)
    h, _ = np.histogram(v, edges)
    hs = gaussian_filter1d(h.astype(float), 2.0)
    peaks, _ = find_peaks(hs, distance=max(1, int(min_sep / bin_w)), prominence=0.05 * hs.max())
    if len(peaks) == 0:
        return np.rint(mu).astype(np.int64)
    centers = 0.5 * (edges[peaks] + edges[peaks + 1])
    # Anchor the rank to the nearest integer of the lowest peak so a missing
    # bottom vertebra does not silently relabel the rest.
    base = int(np.rint(centers[0]))
    nearest = np.abs(mu[:, None] - centers[None, :]).argmin(1)
    return base + nearest


def cluster_votes(votes: np.ndarray, radius: float, min_pts: int, max_clusters: int = 32,
                  min_size_fraction: float = 0.3):
    """Greedy mean-shift: densest seed, iterate to the local mean, remove, repeat."""
    remaining = np.arange(len(votes))
    centers, members = [], []
    while len(remaining) >= min_pts and len(centers) < max_clusters:
        sub = votes[remaining]
        keys = np.floor(sub / radius).astype(np.int64)
        uk, inv, cnt = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
        c = sub[inv.reshape(-1) == cnt.argmax()].mean(0)
        tree = KDTree(sub)
        for _ in range(10):
            idx = tree.query_ball_point(c, radius)
            if not idx:
                break
            c_new = sub[idx].mean(0)
            if np.linalg.norm(c_new - c) < 0.1:
                break
            c = c_new
        idx = np.array(tree.query_ball_point(c, radius), dtype=np.int64)
        if len(idx) < min_pts:
            break
        centers.append(c)
        members.append(remaining[idx])
        remaining = np.delete(remaining, idx)
    # Merge clusters that converged onto the same vertebra, then drop small
    # ones: a real vertebra collects votes from all of its visible surface.
    merged = True
    while merged and len(centers) > 1:
        merged = False
        C = np.array(centers)
        D = np.linalg.norm(C[:, None] - C[None], axis=-1) + np.eye(len(C)) * 1e9
        i, j = np.unravel_index(D.argmin(), D.shape)
        if D[i, j] < radius:
            members[i] = np.concatenate([members[i], members[j]])
            centers[i] = votes[members[i]].mean(0)
            del centers[j], members[j]
            merged = True
    if members:
        sizes = np.array([len(m) for m in members])
        keep = sizes >= max(min_pts, min_size_fraction * np.median(sizes))
        centers = [c for c, k in zip(centers, keep) if k]
        members = [m for m, k in zip(members, keep) if k]
    return np.array(centers).reshape(-1, 3), members


def vertebra_centroids(votes: np.ndarray, levels: np.ndarray, weights: np.ndarray | None,
                       radius: float, min_pts: int):
    """One centroid per decoded level: weighted mean-shift seeded at the weighted median.

    Grouping by the level head first, then locating, is far more robust than
    clustering votes blind: the level coordinate is the reliable signal, and
    votes from the ends of a scan are often too spread to cluster on their own.
    Returns (centers (K,3), level (K,), n_votes (K,)), sorted by level.
    """
    w_all = np.ones(len(votes)) if weights is None else np.asarray(weights, float)
    centers, lv, counts = [], [], []
    for L in np.unique(levels[levels >= 0]):
        m = levels == L
        if m.sum() < min_pts:
            continue
        v, w = votes[m], w_all[m] + 1e-6
        c = np.array([_weighted_median(v[:, d], w) for d in range(3)])
        for _ in range(20):
            near = np.linalg.norm(v - c, axis=1) <= radius
            if not near.any():
                break
            c_new = (v[near] * w[near, None]).sum(0) / w[near].sum()
            if np.linalg.norm(c_new - c) < 0.05:
                c = c_new
                break
            c = c_new
        centers.append(c); lv.append(int(L)); counts.append(int(m.sum()))
    return np.array(centers).reshape(-1, 3), np.array(lv, np.int64), np.array(counts, np.int64)


def _weighted_median(x: np.ndarray, w: np.ndarray) -> float:
    o = np.argsort(x)
    cw = np.cumsum(w[o])
    return float(x[o][np.searchsorted(cw, 0.5 * cw[-1])])


def order_consistent(centers: np.ndarray, cluster_levels: np.ndarray) -> bool:
    """Clusters sorted along +z (rostral) must have strictly increasing decoded levels."""
    if len(centers) < 2:
        return True
    o = np.argsort(centers[:, 2])
    lv = cluster_levels[o]
    return bool(np.all(np.diff(lv) > 0))


def ece_brier(p: np.ndarray, y: np.ndarray, bins: int = 15):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    ece, table = 0.0, []
    for b in range(bins):
        m = idx == b
        if m.any():
            conf, acc = p[m].mean(), y[m].mean()
            ece += m.mean() * abs(conf - acc)
            table.append({"bin": f"{edges[b]:.2f}-{edges[b+1]:.2f}", "n": int(m.sum()),
                          "mean_conf": float(conf), "frac_true": float(acc)})
    return float(ece), float(((p - y) ** 2).mean()), table


def _name(label: int) -> str:
    return VERSE_LABELS.get(int(label), str(label))


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def compute_metrics(results: list[dict], cfg: Config) -> dict:
    ec = cfg.eval
    space = part_space(cfg)
    NUM_PARTS, PART_CLASSES = space.num, list(space.names)
    VERT_PARTS = np.array(sorted(space.vertebra_parts))
    conf_mat = np.zeros((NUM_PARTS, NUM_PARTS), np.int64)
    per_level = defaultdict(lambda: defaultdict(list))
    lvl_abs, lvl_z = [], []
    vert_ok = {"round": [], "peaks": []}
    off_by_one = 0
    scan_all_ok = []
    off_err = []
    cen_err, missed, spurious, consistent = [], 0, 0, []
    conf_p, conf_raw, conf_y = [], [], []

    for r in results:
        gt, pr = r["part"], r["part_pred"]
        lab = gt != IGNORE_INDEX
        np.add.at(conf_mat, (gt[lab], pr[lab]), 1)
        vl = r["vert_label"]

        for L in np.unique(vl[lab & (vl > 0)]):
            m = lab & (vl == L)
            per_level[_name(L)]["part_acc"].append(float((gt[m] == pr[m]).mean()))

        # level coordinate
        lm = r["level_mask"].astype(bool)
        if lm.any():
            err = r["level_mu"][lm] - r["level"][lm]
            lvl_abs.append(np.abs(err))
            lvl_z.append(err / np.maximum(r["level_std"][lm], 1e-6))
        is_vert_pred = np.isin(pr, VERT_PARTS)
        decoded = {m: decode_levels(r["level_mu"], is_vert_pred, m) for m in vert_ok}

        sample_ok = True
        for j in np.where(r["gt_valid"])[0]:
            L, gl = int(r["gt_label"][j]), int(r["gt_level"][j])
            m = lm & (vl == L)
            if m.sum() == 0:
                continue
            for meth, dec in decoded.items():
                vals, cnts = np.unique(dec[m], return_counts=True)
                ok = int(vals[cnts.argmax()]) == gl
                vert_ok[meth].append(ok)
                per_level[_name(L)][f"index_ok_{meth}"].append(float(ok))
                if meth == "peaks":
                    sample_ok &= ok
                    if abs(int(vals[cnts.argmax()]) - gl) == 1:
                        off_by_one += 1
            per_level[_name(L)]["level_mae"].append(float(np.abs(r["level_mu"][m] - gl).mean()))
        scan_all_ok.append(sample_ok)

        # centroid offsets
        om = r["offset_mask"].astype(bool)
        if om.any():
            e = np.linalg.norm(r["offset_pred"][om] - r["offset"][om], axis=1)
            off_err.append(e)
            for L in np.unique(vl[om]):
                per_level[_name(L)]["offset_err_mm"].append(float(np.median(e[vl[om] == L])))

        # per-level centroids from votes + order cross-check
        if r["gt_valid"].any():
            votes = r["xyz"][is_vert_pred] + r["offset_pred"][is_vert_pred]
            centers, cl_lv, _ = vertebra_centroids(votes, decoded["peaks"][is_vert_pred],
                                                   r["conf"][is_vert_pred],
                                                   ec.vote_cluster_radius_mm, ec.min_cluster_points)
            gtc = r["gt_centroids"][r["gt_valid"]]
            gtl = r["gt_label"][r["gt_valid"]]
            if len(centers):
                D = np.linalg.norm(centers[:, None] - gtc[None], axis=-1)
                ri, ci = linear_sum_assignment(D)
                good = D[ri, ci] < 2 * ec.vote_cluster_radius_mm
                for a, b in zip(ri[good], ci[good]):
                    cen_err.append(D[a, b])
                    per_level[_name(gtl[b])]["centroid_err_mm"].append(float(D[a, b]))
                missed += len(gtc) - good.sum()
                spurious += len(centers) - good.sum()
                consistent.append(order_consistent(centers, cl_lv))
            else:
                missed += len(gtc)

        conf_p.append(r["conf"]); conf_raw.append(1 / (1 + np.exp(-r["conf_logit"])))
        conf_y.append(r["conf_target"])

    # ---- aggregate
    tp = np.diag(conf_mat).astype(float)
    support = conf_mat.sum(1)
    union = conf_mat.sum(0) + support - tp
    iou = {PART_CLASSES[c]: float(tp[c] / union[c]) for c in range(NUM_PARTS) if support[c] > 0}
    out: dict = {
        "n_samples": len(results),
        "part": {
            "accuracy": float(tp.sum() / max(conf_mat.sum(), 1)),
            "mIoU": float(np.mean(list(iou.values()))) if iou else float("nan"),
            "iou": iou,
            "confusion": conf_mat.tolist(),
            "classes": PART_CLASSES,
        },
    }
    if lvl_abs:
        a, z = np.concatenate(lvl_abs), np.concatenate(lvl_z)
        out["level"] = {
            "point_mae": float(a.mean()), "point_p95": float(np.percentile(a, 95)),
            "z_within_1sigma": float((np.abs(z) < 1).mean()),   # 0.683 if calibrated
            "z_within_2sigma": float((np.abs(z) < 2).mean()),   # 0.954 if calibrated
            "vertebra_index_acc_peaks": float(np.mean(vert_ok["peaks"])) if vert_ok["peaks"] else None,
            "vertebra_index_acc_round": float(np.mean(vert_ok["round"])) if vert_ok["round"] else None,
            "n_vertebrae": len(vert_ok["peaks"]),
            "off_by_one_vertebrae": int(off_by_one),
            "samples_all_correct": float(np.mean(scan_all_ok)),
            "acceptance_target": 0.99,
        }
    if off_err:
        e = np.concatenate(off_err)
        out["offset"] = {"median_mm": float(np.median(e)), "p95_mm": float(np.percentile(e, 95))}
    if cen_err or missed:
        ce = np.array(cen_err) if cen_err else np.array([np.nan])
        out["centroid_votes"] = {
            "median_err_mm": float(np.nanmedian(ce)), "p95_err_mm": float(np.nanpercentile(ce, 95)),
            "worst_err_mm": float(np.nanmax(ce)), "missed": int(missed), "spurious": int(spurious),
            "order_consistent_samples": float(np.mean(consistent)) if consistent else None,
        }
    if conf_p:
        p, pr_, y = np.concatenate(conf_p), np.concatenate(conf_raw), np.concatenate(conf_y)
        ece, brier, table = ece_brier(p, y, ec.conf_bins)
        ece_raw, brier_raw, _ = ece_brier(pr_, y, ec.conf_bins)
        out["confidence"] = {"ece": ece, "brier": brier, "ece_uncalibrated": ece_raw,
                             "brier_uncalibrated": brier_raw, "positive_rate": float(y.mean()),
                             "reliability": table}

    lv = {}
    for name in sorted(per_level, key=lambda n: -VERSE_ORDER[[k for k, v in VERSE_LABELS.items()
                                                               if v == n][0]]):
        d = per_level[name]
        row = {"n": len(d.get("part_acc", []))}
        for k, vals in d.items():
            row[k] = float(np.mean(vals))
            if k in ("centroid_err_mm", "offset_err_mm", "level_mae"):
                row[k + "_worst"] = float(np.max(vals))
        lv[name] = row
    out["per_level"] = lv
    return out


def score(metrics: dict) -> float:
    """Model-selection score: part mIoU plus vertebra index accuracy."""
    s = metrics["part"]["mIoU"]
    acc = metrics.get("level", {}).get("vertebra_index_acc_peaks")
    return float(s + (acc if acc is not None else 0.0))


def write_report(metrics: dict, out_dir: str | Path, name: str = "eval") -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{name}.json", "w") as f:
        json.dump(metrics, f, indent=2)
    L = [f"# Labelling network -- {name}", "", f"Samples: {metrics['n_samples']}", ""]
    p = metrics["part"]
    L += ["## Part labels", f"accuracy {p['accuracy']:.3f}, mIoU {p['mIoU']:.3f}", "",
          "| class | IoU |", "|---|---|"] + [f"| {k} | {v:.3f} |" for k, v in p["iou"].items()]
    if "level" in metrics:
        lv = metrics["level"]
        L += ["", "## Level coordinate",
              f"- relative index correct per vertebra (peaks): **{lv['vertebra_index_acc_peaks']:.4f}**"
              f" over {lv['n_vertebrae']} vertebrae (target > 0.99)",
              f"- relative index correct per vertebra (round): {lv['vertebra_index_acc_round']:.4f}",
              f"- off-by-one vertebrae: {lv['off_by_one_vertebrae']}",
              f"- samples with every vertebra correct: {lv['samples_all_correct']:.3f}",
              f"- point MAE {lv['point_mae']:.3f}, p95 {lv['point_p95']:.3f}",
              f"- sigma coverage: {lv['z_within_1sigma']:.3f} within 1 (0.683 ideal), "
              f"{lv['z_within_2sigma']:.3f} within 2 (0.954 ideal)"]
    if "offset" in metrics:
        L += ["", "## Centroid offsets",
              f"- per-point error median {metrics['offset']['median_mm']:.2f} mm, "
              f"p95 {metrics['offset']['p95_mm']:.2f} mm"]
    if "centroid_votes" in metrics:
        c = metrics["centroid_votes"]
        L += [f"- clustered centroid error median {c['median_err_mm']:.2f} mm, "
              f"p95 {c['p95_err_mm']:.2f} mm, worst {c['worst_err_mm']:.2f} mm",
              f"- missed {c['missed']}, spurious {c['spurious']}, "
              f"order-consistent samples {c['order_consistent_samples']}"]
    if "confidence" in metrics:
        c = metrics["confidence"]
        L += ["", "## Confidence",
              f"- ECE {c['ece']:.4f} (uncalibrated {c['ece_uncalibrated']:.4f}), "
              f"Brier {c['brier']:.4f} (uncalibrated {c['brier_uncalibrated']:.4f}), "
              f"positive rate {c['positive_rate']:.3f}"]
    cols = ["n", "part_acc", "index_ok_peaks", "level_mae", "offset_err_mm", "centroid_err_mm",
            "centroid_err_mm_worst"]
    L += ["", "## Per level (caudal first)", "", "| level | " + " | ".join(cols) + " |",
          "|---" * (len(cols) + 1) + "|"]
    for lvl_name, row in metrics["per_level"].items():
        L.append(f"| {lvl_name} | " + " | ".join(
            (f"{row[c]:.3f}" if isinstance(row.get(c), float) else str(row.get(c, "-"))) for c in cols)
            + " |")
    path = out_dir / f"{name}.md"
    path.write_text("\n".join(L) + "\n")
    return path
