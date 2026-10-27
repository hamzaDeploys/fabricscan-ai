# Data Preprocessing Pipeline

This document describes every step between the raw images and the tensors the models train on, with the reason for
each step and the exact setting used. The raw images are the author's own phone photos (see
[DATA_COLLECTION_GUIDE.md](DATA_COLLECTION_GUIDE.md)) plus the MVTec AD carpet images imported by
`scripts/import_mvtec_carpet.py` (see [dataset/README.md](../dataset/README.md)). No Kaggle data is used.

```
dataset/<class>/<sample>/*.jpg                 raw phone photos (any size, JPEG/PNG/HEIC)
   │  python scripts/clean_dataset.py          STEP 1: validate, orient, strip EXIF, crop, resize,
   │                                                   quality flags, duplicates, sample groups
   ▼
data/processed/<class>/<group>__<name>.jpg     256 x 256 RGB JPEG, no metadata
   │  python scripts/split_dataset.py          STEP 2: group-aware, stratified 70/15/15 + leakage audit
   ▼
data/splits/{train,val,test}/<class>/*.jpg
   │  fabricscan/data.py (inside notebooks)    STEP 3: tf.data loading, resize to 224, augmentation (train only)
   ▼
model input: float32 RGB 224 x 224, values 0-255   → normalisation happens INSIDE the model
```

Supporting tools: `scripts/dataset_report.py` (tables and figures), `scripts/preview_augmentation.py` (augmentation
figure) and `scripts/smoke_test.py` (runs the whole pipeline on code-generated test images).

---

## Step 1: Cleaning (`scripts/clean_dataset.py`, `fabricscan/cleaning.py`)

| # | Step | What happens | Why | Setting |
|---|---|---|---|---|
| 1 | Discover | Walks `dataset/<class>/`. Names starting with `_` or `.` are ignored (e.g. `_rejected/`). Unknown folders are reported. | Lets you park doubtful photos without deleting them. | — |
| 2 | Integrity check | `verify()` plus a full decode. Unreadable or truncated files are removed. HEIC is read through `pillow-heif`. | A single broken file would crash training hours later. | reason `corrupt_or_unreadable` |
| 3 | File type | Non-image files are skipped and listed. | Videos and notes end up in phone exports. | reason `unsupported_file_type` |
| 4 | Orientation | The EXIF orientation tag is applied to the pixels. | Phones store portrait photos sideways and rely on the tag. Once metadata is stripped, the image would otherwise appear rotated. | — |
| 5 | Colour mode | Converted to RGB. Transparency is composited onto white. | The models expect exactly 3 channels. | — |
| 6 | Minimum size | Short side must be ≥ 256 px. | Rejects thumbnails and screenshots. | `--min-side 256`, reason `too_small` |
| 7 | Square crop | Centre crop to 1:1. Non-square inputs get the flag `center_cropped`. | The model input is square. Cropping avoids distorting the weave geometry, which stretching would do. The guide asks for 1:1 capture so nothing is lost. | — |
| 8 | Quality metrics | Measured on a 512 px version: **sharpness** = variance of the Laplacian, **brightness** = mean grey level, **glare** = share of pixels ≥ 250. | Blur, darkness and glare hide defects or create fake ones. | flags `blurry` (< 15), `too_dark` (< 35), `overexposed` (> 25 %) |
| 9 | Resize | 256 × 256, Lanczos filter. | A uniform size; 256 (not 224) leaves room for the random-crop zoom augmentation. | `--size 256` |
| 10 | Metadata removal | Re-encoded as JPEG quality 95 with **no** EXIF. | Privacy (GPS location, phone model) in a public repository, and smaller files. | — |
| 11 | Sample groups | Group = the sample sub-folder (`s001`, …). For flat folders, photos taken ≤ 30 s apart (EXIF time) form one group. | Needed for the leakage-free split in step 2. | `--time-gap 30` |
| 12 | Exact duplicates | SHA-256 of the file bytes. Same class → the later copy is removed. **Different classes → both copies removed** (the same file cannot have two labels). | Duplicates over-weight an image; conflicting labels poison training. | reasons `exact_duplicate_of`, `label_conflict_with` |
| 13 | Near duplicates | RMSE between 32 × 32 thumbnails (0–255 scale); see the table below. | Catches re-saved, re-compressed or resized copies that SHA-256 misses. | `--duplicate-rmse 3.0`, `--burst-rmse 1.0` |

**Near-duplicate rules**

| Pair | Rule | Action |
|---|---|---|
| Same class, different samples | RMSE ≤ 3.0 | Later copy removed (`near_duplicate_of`). It could otherwise sit in both train and test. |
| Same class, same sample | RMSE ≤ 1.0 | Later copy removed (burst or re-saved copy). Rotated or re-lit shots of the same sample are **kept**. |
| Different classes | RMSE ≤ 3.0 | Both **kept** and flagged `check_label` for a manual look. A before/after pair of the same spot, shot from a stand, is legitimate and useful. |

*Calibration:* on the smoke-test images, re-saved or resized copies scored 0.6–0.7, while rotated or re-lit shots of
the same sample, and images of different samples, scored 11 or more. A threshold of 3.0 therefore leaves a wide
margin on both sides.

*Design note (problem faced):* the first version used a 64-bit perceptual hash confirmed by a high-pass texture
correlation. It failed to detect a re-saved copy of a smooth, almost texture-free image (a soft colour-bleeding
halo), because the high-pass signal of such an image is mostly noise. It also would have deleted legitimate
before/after pairs as "label conflicts". The thumbnail-RMSE rules above fixed both cases. The smoke test plants these
exact situations so they cannot silently regress.

**Outputs**

- `data/processed/<class>/<group>__<original name>.jpg`
- `data/reports/clean_manifest.csv`: one row per input file (status, reason, flags, size, sharpness, brightness,
  glare, capture time, SHA-256), the full audit trail
- `data/reports/clean_summary.json`: counts per class, removal reasons and warnings (classes under 150 images,
  imbalance above 1.5×, fewer than 3 sample groups, groups with more than 8 shots)
- `data/reports/flagged/<flag>/`: copies of flagged images for a quick visual review. Flagged images stay in the
  dataset unless you re-run with `--drop-flagged`.

---

## Step 2: Split (`scripts/split_dataset.py`, `fabricscan/splitting.py`)

| Property | How |
|---|---|
| **Ratios** | 70 % train / 15 % validation / 15 % test, measured in images |
| **Stratified** | Each class is split on its own, so every split keeps the class balance |
| **Group-aware** | All shots of one physical sample go to the same split. Otherwise near-identical shots of one defect would sit in train and test, and the test accuracy would be inflated (**data leakage**). |
| **Algorithm** | Groups are shuffled with the seed, sorted largest-first, and each is given to the split furthest below its target share. Equal-size groups land on the exact ratio. |
| **Reproducible** | `--seed 42` (default). The same seed and input always give the same split; `split_manifest.csv` records it. |
| **Minimum** | Every class needs at least 3 sample groups, so each split gets one |
| **Leakage audit** | After copying, re-checks that (a) no sample group appears in two splits and (b) no pair of images in different splits is a near duplicate (thumbnail RMSE ≤ 3.0). The result is printed and saved in `split_summary.json`. |

Outputs: `data/splits/{train,val,test}/<class>/`, `data/reports/split_manifest.csv`, `data/reports/split_summary.json`.
`--zip` also writes `data/fabricscan_splits.zip` for uploading to Google Drive or Colab.

---

## Step 3: Model input pipeline (`fabricscan/data.py`)

### Loading and preprocessing (all splits)

1. Read and decode the JPEG to uint8 RGB, then resize to 256 × 256 (a no-op for cleaned images).
2. Cache the decoded uint8 images in memory (about 200 MB per 1,000 images) so the disk is read once.
3. **Validation, test and the website:** resize the whole square image to **224 × 224** (bilinear, antialiased).
   No randomness, so evaluation is repeatable. The browser does exactly the same: centre-crop to a square, then
   resize to 224.
4. **Normalisation is inside each model**, not in the pipeline:
   - custom CNN: `Rescaling(1/255)` → values in [0, 1]
   - MobileNetV2: `Rescaling(1/127.5, offset=-1)` → values in [-1, 1] (what its ImageNet weights expect)

   The exported ONNX model therefore takes raw 0–255 pixels. Python and JavaScript cannot drift apart on
   preprocessing, which is a classic source of "it works in the notebook but not on the website" bugs.

### Augmentation (training split only)

Applied on the fly, so every epoch sees new variations. Applied in this order:

| Transform | Range | Why it is realistic for fabric photos |
|---|---|---|
| Small rotation | ±15°, reflect padding | The phone is never perfectly aligned. Reflect padding avoids black corners the model could learn as an artefact. |
| Zoom | random crop of 80–100 % of the side, resized to 224 | The capture distance varies (20–35 cm). At most 20 % of a side is cut, and the guide keeps defects ≥ 10 % from the border, so the defect stays in frame. |
| Flips | horizontal and vertical, p = 0.5 each | A fabric has no left/right or up/down. |
| 90° rotations | 0°, 90°, 180°, 270° | The same reason; lossless (no interpolation). |
| Brightness | × U(0.8, 1.2) | Daylight vs. indoor lamps |
| Contrast | × U(0.85, 1.15) around the mean | Phone tone-mapping and lighting differences |
| Gaussian noise | σ ~ U(0, 8) on the 0–255 scale | Sensor noise in dim light |

**Deliberately not used:**

- **Hue or saturation shifts:** colour *is* the evidence for `stain`. Shifting it would make
  the label wrong.
- **Strong blur:** it would erase thin broken threads.
- **Cutout / random erasing:** it could erase the defect itself, producing a "defect" image that shows clean fabric.
- **Shear or perspective warps beyond ±15°:** the capture protocol keeps tilt small, so these would only add
  unrealistic images.

`results/dataset/augmentation_preview.png` (made by `scripts/preview_augmentation.py`) shows the effect.

### Reproducibility

- Every random op is *stateless* (`tf.random.stateless_*`) and takes its seed from
  `tf.data.Dataset.random(seed=42, rerandomize_each_iteration=True)`. Each epoch gets fresh augmentations, yet two
  runs with the same seed see the same sequence. The unit tests check both properties.
- Shuffling is seeded. `fabricscan.seed.set_global_seed()` seeds Python, NumPy and TensorFlow, and can enable
  deterministic TensorFlow kernels.

### Class imbalance

`fabricscan.data.class_weights()` returns "balanced" weights, `n_total / (n_classes × n_class)`, which are passed
to `model.fit(class_weight=...)`. Each class then contributes equally to the loss even when, for example,
`broken_thread` ends up with fewer photos than `good`.

---

## Field test set (optional)

Real-world photos in `field_test/<class>/` go through the same cleaning step, into a separate folder, and are
**never** split, trained on or used for tuning:

```
python scripts/clean_dataset.py --input field_test --output data/field_test_processed --reports data/reports/field_test
```

---

## Commands at a glance

```
python scripts/clean_dataset.py              # STEP 1  dataset/        -> data/processed/
python scripts/split_dataset.py --zip        # STEP 2  data/processed/ -> data/splits/  (+ zip for Colab)
python scripts/dataset_report.py             #         tables + figures -> results/dataset/
python scripts/preview_augmentation.py       #         augmentation figure -> results/dataset/
python -m pytest                             #         unit tests
python scripts/smoke_test.py                 #         whole pipeline on code-generated images (data/_smoke_test/)
```

The smoke test uses images drawn by code (`fabricscan/synthetic.py`) to prove the pipeline works before the real
photos exist. They live only in `data/_smoke_test/` and are never used to train or evaluate the submitted models.
