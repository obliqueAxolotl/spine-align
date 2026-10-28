"""Smoke and invariant tests on the phantom spine. Run: pytest -q tests"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from phantom import make_phantom, write_phantom_stl, write_phantom_totalsegmentator  # noqa: E402

from spine_labelnet.config import load_config
from spine_labelnet.constants import PART
from spine_labelnet.data.verse import load_mask, make_split, subject_of

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def workdir(tmp_path_factory):
    d = tmp_path_factory.mktemp("smoke")
    for i in range(6):
        make_phantom(d / "raw" / "Masks" / f"sub-verse{200 + i:03d}_ct.nii.gz", seed=i)
    return d


@pytest.fixture(scope="session")
def cfg(workdir):
    c = load_config(ROOT / "configs" / "smoke.yaml", [
        f"data.raw_dir={workdir / 'raw'}", f"data.cache_dir={workdir / 'cache'}",
        f"data.split_file={workdir / 'splits.json'}", f"train.out_dir={workdir / 'run'}",
        "train.epochs=2"])
    return c


@pytest.fixture(scope="session")
def caches(cfg, workdir):
    from spine_labelnet.data.preprocess import run_preprocess
    run_preprocess(cfg, workers=1)
    # Six subjects are too few for hash fractions to fill every split; fix it.
    subs = sorted(json.load(open(cfg.data.split_file)).values(), key=len)
    allsubs = sorted(s for v in subs for s in v)
    json.dump({"train": allsubs[:4], "val": allsubs[4:5], "test": allsubs[5:]},
              open(cfg.data.split_file, "w"))
    return sorted((workdir / "cache").glob("*.npz"))


def test_affine_uses_qform(workdir):
    p = next((workdir / "raw" / "Masks").glob("*.nii.gz"))
    img = nib.load(str(p))
    assert int(img.header["sform_code"]) == 0 and int(img.header["qform_code"]) == 1
    m = load_mask(p)
    assert m.axcodes == ("R", "A", "S")
    np.testing.assert_allclose(m.spacing, [0.9, 0.9, 1.5], atol=1e-4)


def test_split_is_by_subject():
    scans = ["sub-verse500_split-verse290_ct", "sub-verse500_split-verse291_ct", "sub-verse501_ct"]
    subs = [subject_of(s) for s in scans]
    assert subs[0] == subs[1] == "sub-verse500"
    split = make_split(subs, 0.3, 0.3, seed=1)
    owners = [k for k, v in split.items() if "sub-verse500" in v]
    assert len(owners) == 1


def _zip_layout(tmp_path, entries):
    import zipfile
    src = tmp_path / "src.nii.gz"
    if not src.exists():
        make_phantom(src, labels=(20, 21, 22, 23, 24), seed=3)
    zpath = tmp_path / f"part{len(list(tmp_path.glob('*.zip')))}.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        for name in entries:
            zf.write(src, name)
    return zpath


@pytest.mark.parametrize("entries,expect_subject", [
    # same name in Images/ and Masks/, nested under a top folder
    (["VerSe-Lite/Images/sub-verse004_ct.nii.gz", "VerSe-Lite/Masks/sub-verse004_ct.nii.gz",
      "VerSe-Lite/README.md", "__MACOSX/VerSe-Lite/Masks/._sub-verse004_ct.nii.gz"], "sub-verse004"),
    # VerSe's native naming: mask and image names differ
    (["Images/sub-verse506_dir-iso_ct.nii.gz", "Masks/sub-verse506_dir-iso_seg-vert_msk.nii.gz"],
     "sub-verse506"),
    # no sub- part: each scan must be its own subject
    (["images/VerSe_001.nii.gz", "masks/VerSe_001.nii.gz"], "VerSe_001"),
])
def test_archive_layouts(tmp_path, entries, expect_subject):
    from spine_labelnet.data.check import check_data
    from spine_labelnet.data.verse import discover_scans, extract_zip, find_image
    z = _zip_layout(tmp_path, entries)
    raw = tmp_path / "raw"
    counts = extract_zip(z, raw, include_images=True)
    assert counts["mask"] == 1 and counts["image"] == 1
    recs = discover_scans(raw)
    assert len(recs) == 1 and recs[0].subject_id == expect_subject
    assert find_image(recs[0]) is not None
    lines, ok = check_data(raw, list(range(8, 26)), sample=1, need_images=True)
    assert ok, lines


def test_archive_without_masks_fails_loudly(tmp_path):
    from spine_labelnet.data.verse import extract_zip
    z = _zip_layout(tmp_path, ["data/sub-verse004_ct.nii.gz"])
    with pytest.raises(RuntimeError, match="no NIfTI files under a Masks/ folder"):
        extract_zip(z, tmp_path / "raw", include_images=False)


def test_parcellation_is_bilateral(caches):
    z = np.load(caches[0])
    part = z["part"]
    for a, b in [("lamina_left", "lamina_right"), ("transverse_left", "transverse_right"),
                 ("facet_left", "facet_right")]:
        na, nb = (part == PART[a]).sum(), (part == PART[b]).sum()
        assert na > 0 and nb > 0 and 0.6 < na / nb < 1.6, (a, na, nb)
    assert (part == PART["spinous_process"]).sum() > 0
    assert z["truncated"][-1] and not z["truncated"][0]       # phantom top is cut by the FOV


def test_sampler_targets(caches, cfg):
    from spine_labelnet.classes import part_space
    from spine_labelnet.data.sample import sample_surface
    z = dict(np.load(caches[0]))
    rng = np.random.default_rng(0)
    for _ in range(5):
        s = sample_surface(z, cfg.sample, part_space(cfg), rng)
        lv = np.unique(s["level"][s["level_mask"]]).astype(int)
        assert lv[0] == 0 and np.all(np.diff(lv) == 1)         # relative index, contiguous
        g = s["gt_valid"]
        # rostral vertebrae have larger index and larger z
        order = np.argsort(s["gt_level"][g])
        assert np.all(np.diff(s["gt_centroids"][g][order][:, 2]) > 0)
        # offsets land exactly on the centroid of their vertebra
        m = s["offset_mask"]
        votes = s["xyz"][m] + s["offset"][m]
        for j in np.where(g)[0]:
            mv = s["vert_label"][m] == s["gt_label"][j]
            if mv.any():
                np.testing.assert_allclose(votes[mv].mean(0), s["gt_centroids"][j], atol=1e-2)
        assert s["xyz"].shape == (cfg.sample.num_points, 3)
        assert (s["part"] == PART["vertebral_body"]).any()       # full surface, body included


def test_extra_bone_classes(tmp_path):
    """All non-vertebral bone is other_bone by default; a config change splits it, no reprocessing."""
    from spine_labelnet.classes import build_part_space, part_space
    from spine_labelnet.data.extra_bone import BONE_CODE, build_extra_masks
    from spine_labelnet.data.preprocess import run_preprocess
    from spine_labelnet.data.sample import sample_surface
    mask = make_phantom(tmp_path / "raw" / "Masks" / "sub-verse900_ct.nii.gz",
                        labels=(17, 18, 19, 20, 21, 22, 23, 24), bottom_margin=60,
                        half_width=100, seed=1)
    write_phantom_totalsegmentator(mask, tmp_path / "ts" / "sub-verse900_ct")
    res = build_extra_masks(str(tmp_path / "raw"), str(tmp_path / "extra"), ts_dir=str(tmp_path / "ts"))
    assert res["written"] == 1 and not res["failed"]
    c = load_config(ROOT / "configs" / "smoke.yaml", [
        f"data.raw_dir={tmp_path / 'raw'}", f"data.cache_dir={tmp_path / 'cache'}",
        f"data.split_file={tmp_path / 's.json'}", f"data.extra_masks_dir={tmp_path / 'extra'}"])
    run_preprocess(c, workers=1)
    z = dict(np.load(tmp_path / "cache" / "sub-verse900_ct.npz"))

    # cache keeps exact identity: ribs 10-12 both sides, sacrum, both hips
    ids = set(z["bone_id"][z["bone_id"] > 0].tolist())
    expect = {BONE_CODE[n] for n in ["sacrum", "hip_left", "hip_right"]
              + [f"rib_{s}_{k}" for s in ("left", "right") for k in (10, 11, 12)]}
    assert ids == expect
    assert np.all(z["vert_label"][z["bone_id"] > 0] == 0)       # never owns a vertebra label

    def classes_seen(space):
        rng, seen = np.random.default_rng(0), set()
        for _ in range(20):
            s = sample_surface(z, c.sample, space, rng)
            extra = s["part"] >= len(space.vertebra_parts)
            assert not s["level_mask"][extra].any() and not s["offset_mask"][extra].any()
            seen |= {space.names[i] for i in np.unique(s["part"][extra])}
        return seen

    default = part_space(c)
    assert default.names[-1] == "other_bone" and default.num == 11
    assert classes_seen(default) == {"other_bone"}

    split = build_part_space(["rib", "sacrum", "iliac", "other_bone"],
                             [["rib_*", "rib"], ["sacrum", "sacrum"], ["hip_*", "iliac"],
                              ["*", "other_bone"]])
    assert classes_seen(split) == {"rib", "sacrum", "iliac"}
    bilateral = build_part_space(["rib_left", "rib_right", "other_bone"],
                                 [["rib_left_*", "rib_left"], ["rib_right_*", "rib_right"],
                                  ["*", "other_bone"]])
    assert bilateral.lr_swap[bilateral.index("rib_left")] == bilateral.index("rib_right")


def test_model_forward_backward(cfg):
    torch = pytest.importorskip("torch")
    from spine_labelnet.losses import compute_losses
    from spine_labelnet.model import SpineLabelNet
    NUM_PARTS = 11
    model = SpineLabelNet(cfg.model, NUM_PARTS)
    B, N = 2, cfg.sample.num_points
    xyz = torch.randn(B, N, 3) * 40
    nrm = torch.nn.functional.normalize(torch.randn(B, N, 3), dim=-1)
    out = model(xyz, nrm)
    assert out["part"].shape == (B, N, NUM_PARTS) and out["offset"].shape == (B, N, 3)
    batch = {"part": torch.randint(0, NUM_PARTS, (B, N)), "level": torch.rand(B, N) * 4,
             "level_mask": torch.ones(B, N, dtype=torch.bool), "offset": torch.randn(B, N, 3),
             "offset_mask": torch.ones(B, N, dtype=torch.bool)}
    losses = compute_losses(out, batch, cfg.loss, None, conf_active=True)
    losses["loss"].backward()
    assert all(torch.isfinite(v) for v in losses.values())


def test_train_evaluate_calibrate_predict(caches, cfg, workdir):
    torch = pytest.importorskip("torch")
    from spine_labelnet.calibrate import calibrate
    from spine_labelnet.cli import _loader
    from spine_labelnet.evaluate import compute_metrics, run_inference, write_report
    from spine_labelnet.predict import predict_file
    from spine_labelnet.train import load_checkpoint, train

    best = train(cfg)
    assert best.exists()
    model, c2, _ = load_checkpoint(best)
    res = run_inference(model, _loader(c2, "val", 2, 20_000), torch.device("cpu"))
    fitted = calibrate(model, res)
    assert fitted["conf_temperature"] > 0 and fitted["level_std_scale"] > 0
    res = run_inference(model, _loader(c2, "test", 2, 30_000), torch.device("cpu"))
    metrics = compute_metrics(res, c2)
    report = write_report(metrics, workdir / "run", "eval_test")
    assert report.name == "eval_test.md" and "Per level" in report.read_text()
    mask = next((workdir / "raw" / "Masks").glob("*.nii.gz"))
    stl = write_phantom_stl(mask, workdir / "scan.stl")
    for src in (stl, mask):                       # STL surface, and a volume thresholded at 0.5
        c2.data.ct_bone_threshold_hu = 0.5
        out = predict_file(model, c2, str(src), str(workdir / f"{src.stem}_labels.npz"),
                           torch.device("cpu"))
        assert len(out["part"]) == len(out["points"]) and len(out["vote_centroids"]) > 0


def test_default_yaml_matches_builtin_defaults():
    """configs/default.yaml must parse and agree with the dataclass defaults."""
    from spine_labelnet.config import Config
    c, d = load_config(ROOT / "configs" / "default.yaml").to_dict(), Config().to_dict()
    c["data"]["levels"], d["data"]["levels"] = sorted(c["data"]["levels"]), sorted(d["data"]["levels"])
    assert c == d
