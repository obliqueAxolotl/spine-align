# spine-labelnet

Component 1 of the Spine Atlas Overlay pipeline: the **labelling network**.
It takes a bone surface extracted from CT and says what each point is. It
never predicts a pose. The kinematic solver (Component 2) turns these
per-point targets into geometry.

There is no ultrasound simulation. Training samples are the full CT bone
surface from VerSe-Lite segmentations, cut into windows of contiguous
vertebrae, with only geometric augmentation (small rotations, left/right
mirroring).

This is a first draft that runs end to end, not a tuned system. Values that
are guesses are marked `provenance: placeholder` in `configs/default.yaml`.

## What the network predicts

| Head | Output | Loss |
|---|---|---|
| Part label | 10 vertebra parts (spinous process, L/R lamina, L/R transverse process, L/R facet, L/R pedicle, vertebral body) + configurable non-vertebral classes, by default one: `other_bone` | class-weighted cross-entropy, ignore-index |
| Level coordinate | mean and log-variance of the relative index (lowest in-view vertebra = 0, increasing rostrally) | Gaussian NLL |
| Centroid offset | mm vector from the point to its vertebra's full-mask centroid | smooth L1 |
| Confidence | P(the point's predicted part and level are both correct) | BCE, then temperature-scaled |

These are what the solver needs: part + level give correspondence (which
atlas patch each point belongs to), clustered centroid votes give one
translation anchor per vertebra and the vertebra order, and calibrated
confidence gives the data-term weights.

The handoff's 11 classes were written for dorsal-only ultrasound. CT sees the
whole vertebra, so body and pedicles are classes here and the ultrasound
"artifact" class is gone.

## Pipeline

```
VerSe-Lite masks (NIfTI, RAS+, qform)
   │  preprocess: marching cubes per vertebra -> surface points + normals
   │              full-mask centroid, vertebra frame, rule-based part labels
   ▼
per-scan cache (.npz)  +  manifest.csv  +  splits.json (by subject)
   │  SpineSurfaceDataset: each item is a window of 3-7 contiguous vertebrae,
   │    partial vertebrae at the ends, small rotations, L/R mirroring
   ▼
PointNet++ (pure PyTorch) + global scan feature -> 4 heads
   │  train -> calibrate (temperature, level-sigma scale) -> evaluate
   ▼
predict: CT volume / STL / npz -> per-point part, level, confidence, vote centroids
```

## Quick start

```bash
pip install -e ".[download,dev]"

# 1. Data: VerSe-Lite is ~49 GB as five zips. Masks only by default;
#    add --images if you will run TotalSegmentator (step 1b).
spine-labelnet download --out data/verse-lite --images

# 1a. Check the extracted files against what the package expects (seconds)
spine-labelnet check-data -c configs/default.yaml --images

# 1b. Optional but recommended: labels for non-vertebral bone (see below)
pip install TotalSegmentator
spine-labelnet extra-bone -c configs/default.yaml -o data.extra_masks_dir=data/extra_bone --run

# 2. Surface caches, manifest and a subject-level split
#    (pass the same -o data.extra_masks_dir=... so the extra bone is included)
spine-labelnet preprocess -c configs/default.yaml --workers 8
spine-labelnet inventory  -c configs/default.yaml      # scans per level

# 3. Look before you train: coloured PLY samples (open in 3D Slicer / CloudCompare)
spine-labelnet preview -c configs/default.yaml --n 10 --out preview/

# 4. Train, calibrate, evaluate
spine-labelnet train -c configs/default.yaml -o train.out_dir=runs/v0
spine-labelnet calibrate --ckpt runs/v0/best.pt          # writes runs/v0/calibrated.pt
spine-labelnet evaluate  --ckpt runs/v0/calibrated.pt --split val
spine-labelnet evaluate  --ckpt runs/v0/calibrated.pt --split test   # only when you mean it

# 5. Inference
spine-labelnet predict --ckpt runs/v0/calibrated.pt ct.nii.gz --out-dir predictions/
```

**About the VerSe-Lite files.** The package was written from the dataset
card, which describes five `data-partNNN.zip` archives holding `Images/` and
`Masks/` (RAS+, uint16 masks on the image grid, 325 scans), but not the file
names. `download` extracts any NIfTI under a `Masks/` (and with `--images`,
`Images/`) folder at any depth, any case, skips `__MACOSX` entries, and stops
with the first archive entries listed if it finds no masks. Images are paired
with masks by identical name, or by VerSe's native naming
(`..._ct` vs `..._seg-vert_msk`). Subjects come from the `sub-verseNNN` part of
the name; a name without one is treated as its own subject. Run `check-data`
first: it reports anything the real files do differently.

`predict` accepts a CT volume (`.nii`/`.nii.gz`, bone taken as everything at
or above `data.ct_bone_threshold_hu`, default 200 HU; use 0.5 for a
segmentation mask), an `.stl` surface, or an `.npz` with `points` and
`normals`. It writes per-point results plus a coloured `.ply`.

Any config value can be overridden on the command line with
`-o section.key=value`. A CPU-sized configuration for the phantom lives in
`configs/smoke.yaml`; `pytest -q tests` runs the whole chain on synthetic
phantom spines in about a minute.

## Non-vertebral bone

VerSe segments vertebrae only, but a thresholded CT, and later an ultrasound
surface scan, also contain ribs, the sacrum, the iliac crest and more. If the
network never sees that bone labelled, it assigns it to the nearest vertebra
class. For now **every non-vertebral bone is one class, `other_bone`**, with
the data kept so that classes can be split out later without reprocessing.

1. `spine-labelnet extra-bone --run` runs TotalSegmentator's `total` task on
   each VerSe-Lite CT image (restricted to the listed structures with
   `roi_subset`), or `--ts-dir DIR` reads output you already have, laid out as
   `DIR/<scan_id>/<class>.nii.gz`. `data.extra_structures` picks the
   structures (default: sacrum, all 24 ribs, hips, sternum, scapulae,
   clavicles, femora, humeri, S1).
2. It writes one mask per scan in `data.extra_masks_dir` whose values are
   stable codes into `BONE_STRUCTURES` (`data/extra_bone.py`), so every bone
   keeps its exact identity. Structures are matched by name, so the
   TotalSegmentator version doesn't matter.
3. `preprocess` extracts one surface per structure and stores that identity
   per point (`bone_id` in the cache). Where TotalSegmentator and VerSe
   disagree about a voxel, VerSe's vertebra wins.
4. When samples are drawn, the `classes` section of the config turns identity
   into a training class. The default sends everything to `other_bone`:

   ```yaml
   classes:
     bone: [other_bone]
     bone_map:
       - ["*", other_bone]
   ```

   To split classes out later, edit both lists and retrain; the caches stay as
   they are. Patterns are matched in order, first match wins:

   ```yaml
   classes:
     bone: [rib, sacrum, iliac, other_bone]
     bone_map:
       - ["rib_*", rib]
       - ["sacrum", sacrum]
       - ["hip_*", iliac]
       - ["*", other_bone]
   ```

   Classes named `*_left` / `*_right` are paired automatically for the mirror
   augmentation. A checkpoint records its class list, so `predict`, `evaluate`
   and `calibrate` always use the classes it was trained with.
5. Non-vertebral bone gets no level and no centroid target, as the handoff
   specifies.
6. `sample.lateral_extent_mm` crops each sample to a random left-right
   half-width (80–150 mm by default) so long ribs don't take over the point
   budget, and a window ending at the lowest vertebra extends into the sacrum
   some of the time (`sample.sacrum_extension_prob`).

What the single class does not do yet: it only covers the structures listed
above. Bone nobody labelled (coccyx, osteophytes, calcified cartilage,
calcified vessels, hardware) never appears in training, so on a thresholded
CT the network has to force it into some class. Training on thresholded CT
surfaces, with unlabelled bone counted as `other_bone`, is the fix when it
matters.

For the ultrasound scans later, the class names and their meaning carry
over unchanged, so a model trained this way can be fine-tuned on annotated
SpineUs surfaces. One thing to settle first, even while ribs are inside
`other_bone`: the labelling guide must say where a rib stops and the
transverse process starts (the costotransverse joint), and annotators must
label exactly that, because that boundary is between a vertebra class and a
non-vertebral one. Ultrasound
will mostly see the dorsal rib surface near the spine and the iliac crest,
not the whole bone.

Check TotalSegmentator's licence terms for the task and weights you use
before building anything commercial on its output.

## Package layout

```
src/spine_labelnet/
  constants.py         part classes, VerSe labels, cranio-caudal order (T13 handled)
  config.py            dataclass config, YAML + overrides
  data/verse.py        download, discovery, qform/affine checks, subject split
  data/check.py        check-data: verify an extracted VerSe-Lite folder
  data/surface.py      marching cubes -> surface samples, centroids, tangents
  data/parcellation.py vertebra frame + rule-based part labels (draft labelling guide)
  classes.py           class space: fixed vertebra parts + configured bone classes
  data/extra_bone.py   TotalSegmentator bone -> combined masks with stable per-structure codes
  data/preprocess.py   cache builder, manifest, inventory
  data/sample.py       vertebra-window sampler and all targets
  data/dataset.py      PyTorch dataset
  data/io.py           CT / STL / npz readers, PLY writer
  model/pointnet2.py   backbone (swap point for a sparse-conv U-Net / PTv3)
  model/labelnet.py    heads + calibration buffers
  losses.py            per-head losses and the confidence target
  train.py             training loop
  calibrate.py         temperature scaling, level-sigma scaling
  evaluate.py          metrics, level decoding, vote clustering, reports
  predict.py           inference -> solver targets
  cli.py
```

## Training regime

- **Data.** Training subjects' CT surface caches; every epoch draws
  `samples_per_spine` fresh windows per spine. Validation uses a fixed-seed
  set of windows of the validation subjects, so epochs are comparable. Splits
  are by subject ID, never by scan or by sample.
- **Optimiser.** AdamW, linear warmup then cosine decay, gradient clipping, AMP on CUDA.
- **Confidence warm-up.** The confidence target depends on the other heads'
  predictions, so that head switches on after `loss.conf_warmup_epochs`.
- **Model selection.** Best checkpoint by part mIoU + per-vertebra level-index accuracy.
- **Learning curve.** Run with `train.train_fraction` at 0.1, 0.25, 0.5, 1.0 and
  plot validation error per head against log sample count.

Memory: `torch.cdist` in the first set-abstraction layer is B × 2048 × 16384
floats; batch 8 needs roughly 10 GB of GPU memory. Lower `sample.num_points`
or `train.batch_size` if needed.

## Validation

`evaluate` writes `eval_<split>.json` and `eval_<split>.md`, reported per level
and never only pooled:

- **Parts:** accuracy, per-class IoU, mIoU, confusion matrix.
- **Level:** relative index correct per vertebra (handoff target > 99 %),
  decoded two ways (rounding, and 1D peak-finding); off-by-one count; samples
  with every vertebra correct; point MAE; and whether the predicted σ covers
  the error at the stated rate (68 % / 95 %).
- **Centroids:** per-point offset error; one centroid per decoded level from
  the points' votes (confidence-weighted mean-shift), with its error (median,
  p95, worst), missed and spurious vertebrae, and the cross-check that
  centroid order along the spine agrees with level order.
- **Confidence:** ECE and Brier score, before and after temperature scaling,
  with a reliability table.

## Placeholders and open items

1. **Part labels are a draft labelling guide.** `data/parcellation.py` applies
   rules in each vertebra's own frame. Write the real guide, tune `data.parcel`
   to match it, and spot-check with `preview`.
2. **Training input is cleaner than inference input.** Training surfaces come
   from expert vertebra masks (plus TotalSegmentator bone); `predict` on a raw
   CT thresholds at a fixed HU, which is noisier, especially around
   osteoporotic bone, and includes bone that no mask labelled.
   With extra-bone training the gap is smaller, but it is still there until
   the network is also trained on thresholded surfaces.
3. **Extra-bone labels are only as good as TotalSegmentator.** Spot-check a
   few scans with `preview`, especially at the rib–transverse process junction
   and where the sacrum meets L5 in transitional anatomy.
4. **Re-posing uses small random rotations, not the atlas rig.** `sample_surface`
   takes a `repose_fn` hook for when the rig exists.
5. **Transitional vertebrae** (L6, T13) are in the label map but not
   oversampled yet.
6. **Backbone.** PointNet++ installs anywhere and is the weakest of the three
   options the handoff names. The heads see only a per-point feature tensor, so
   a sparse-conv U-Net or PTv3 is a change to `model/pointnet2.py` alone.

## Licence of the data

VerSe-Lite is CC BY-SA 4.0. Cite Sekuboyina et al., *Medical Image Analysis*
2021 (doi:10.1016/j.media.2021.102166) and the VerSe-Lite redistribution. The
non-commercial `sub-gl*` scans are already excluded from VerSe-Lite.
