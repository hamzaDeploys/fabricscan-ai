# FabricScan AI

**AI-based fabric defect detection with an automatic inspection report.**

Upload or photograph a piece of fabric or a garment. A convolutional neural network sorts it into **good, hole,
stain or broken thread**, shows where the defect is, estimates its size and severity, and writes the note a quality
inspector would write (copy, PNG or PDF). The model runs entirely in the browser with ONNX Runtime Web, so no photo
leaves the device.

| | |
|---|---|
| **Live website** | **https://fabricscan-ai.vercel.app** |
| **Project report** | [docs/REPORT.md](docs/REPORT.md) |
| **Presentation** | [docs/FabricScan_AI_Presentation.pptx](docs/FabricScan_AI_Presentation.pptx) (14 slides, voice-over in the speaker notes) |
| **Video** | Plays in a popup on the [website](https://fabricscan-ai.vercel.app/#links), or the file: [`fabricscan-presentation.mp4`](web/public/video/fabricscan-presentation.mp4) |
| **Dataset (ZIP)** | [`web/public/downloads/fabricscan-dataset.zip`](web/public/downloads/fabricscan-dataset.zip) |
| **Model weights** | [`weights/`](weights/) |

Deep Learning semester project · **Hamza** · Superior University, Gold Campus · Instructor: **Dr. Munir Ahmed**

## Quick test

```bash
pip install numpy pillow onnxruntime
python predict.py test_samples --eval     # 40 labelled test images: accuracy, per-class F1, confusion matrix
python predict.py path/to/photo.jpg       # any photo
```

`weights/fabricscan.onnx` is the model the website runs (MobileNetV2). Both trained models are also in Keras format
(`weights/*.keras`, TensorFlow 2.20), and `weights/metrics.json` holds the full test results and training history.
[weights/README.md](weights/README.md) describes the input format.

## Results

Test split: 155 images that were never used for training or model selection.

| Model | Accuracy | Macro F1 | Parameters | ONNX size | CPU time / image |
|---|---:|---:|---:|---:|---:|
| MobileNetV2, transfer learning (website default) | **81.3%** | 0.821 | 2.26 M | 8.9 MB | 12.1 ms |
| Custom CNN, trained from scratch | **68.4%** | 0.692 | 1.18 M | 4.7 MB | 5.4 ms |

Per class, MobileNetV2 (transfer learning):

| Class | Precision | Recall | F1 | Test images |
|---|---:|---:|---:|---:|
| Good (No Defect) | 0.86 | 0.81 | 0.83 | 52 |
| Hole | 0.92 | 0.92 | 0.92 | 26 |
| Stain / Oil Spot | 0.90 | 0.65 | 0.76 | 43 |
| Broken Thread / Slub | 0.65 | 0.94 | 0.77 | 34 |

**Field test** (`scripts/field_test.py`): 150 photos of clean garments as people wear them, none of them in the dataset (the live-camera situation): **92% classified as good**. Confusion matrices, training curves and timings: `results/metrics/` and the Performance section of the website.

## Dataset

1,025 images in four classes, 256 × 256 px, built from five public Kaggle datasets:

| Kaggle dataset | Licence | Used for |
|---|---|---|
| [nexuswho/fabric-defects-dataset](https://www.kaggle.com/datasets/nexuswho/fabric-defects-dataset) | CC BY 4.0 | good, hole, stain, broken thread |
| [pulakdroy/fabric-defect-without-augmentation](https://www.kaggle.com/datasets/pulakdroy/fabric-defect-without-augmentation) | CC BY 4.0 | hole, stain (spot), broken thread (missing yarn, slub) |
| [anmol055/clothes-staindefect-detection](https://www.kaggle.com/datasets/anmol055/clothes-staindefect-detection) | CC BY 4.0 | stains on clothes, clean clothes |
| [hyungjung/stitchingnet-dataset](https://www.kaggle.com/datasets/hyungjung/stitchingnet-dataset) | CC BY-NC 4.0 | broken stitch, normal seams |
| [rajkumarl/people-clothing-segmentation](https://www.kaggle.com/datasets/rajkumarl/people-clothing-segmentation) | CC0 | clean shirts, tops and jackets as people wear them |

| Class | Train | Val | Test | Total |
|---|---:|---:|---:|---:|
| Good (No Defect) | 242 | 52 | 52 | 346 |
| Hole | 120 | 26 | 26 | 172 |
| Stain / Oil Spot | 197 | 43 | 43 | 283 |
| Broken Thread / Slub | 156 | 34 | 34 | 224 |
| **Total** | **715** | **155** | **155** | **1025** |

How it was built (`scripts/`):

1. **Import** (`import_sources.py`): labelled defect boxes are cut into square close-ups (2–3 × the box), so the defect
   is visible at the model's 224 px input; clean garments are cropped with the clothing masks. All views of one
   original image form one *group*. `dataset/SOURCES.csv` lists the original file and licence of every image.
2. **Clean** (`clean_dataset.py`): removes corrupt, too small (< 128 px), exact and near duplicates (thumbnail RMSE),
   strips EXIF, crops to a square and resizes to 256 px.
3. **Split** (`split_dataset.py`): 70 / 15 / 15 by group, so near-copies never sit in two splits, followed by a
   leakage audit.
4. **Package** (`package_dataset.py`): the ZIP above, with `SOURCES.csv` and a README.

Two design problems found while testing, and fixed in the data: photos of worn shirts first appeared only in the
stain class, and plain grey cotton mostly in the stain class, so the model learned "shirt photo = stain" and "plain
light cotton = stain". Clean worn garments and the plain grey cotton photos were added to the good class.

## Models

Both models take a 224 × 224 RGB image with raw values 0–255 (normalisation is inside the model) and are trained on
the same split with the same augmentation: rotation ±15°, zoom 80–100 %, flips, 90° turns, brightness ±20 %, contrast
±15 % and sensor noise (no hue changes, colour is evidence for stains). Class weights balance the classes. Batch 16,
seed 42, TensorFlow 2.20 on a CPU.

| | Custom CNN (from scratch) | MobileNetV2 (transfer learning) |
|---|---|---|
| Architecture | 4 conv stages (32-64-128-256), BatchNorm, spatial dropout, L2 1e-5, global max pooling | ImageNet MobileNetV2, global average pooling, dropout, softmax head |
| Training | Adam 1e-3, up to 60 epochs | 12 epochs with the base frozen (Adam 1e-3), then the top 40 layers fine-tuned (Adam 1e-5, BatchNorm frozen) |
| Regularisation | early stopping on validation loss (patience 10), learning rate halved on plateaus | early stopping (patience 6), learning rate halved on plateaus |

Both models are exported to ONNX (opset 17) with two outputs: the class probabilities and the class activation maps.
The export is checked against Keras on test images. The better model (highest macro F1 on the test split) is the
website's default, and a dropdown in the live demo switches to the other one, checking the same photo again.

**Automatic description (in the browser):** the activation map of the predicted class, refined by colour contrast,
gives the defect's zone (3 × 3 grid) and area; per-class area thresholds give the severity, and a small knowledge base
adds likely causes and the recommended action.

## Repository

```
fabricscan/          Python package: data pipeline, models, training, evaluation, ONNX export
scripts/             pipeline steps: import, clean, split, report, train, package, export
weights/             trained models (ONNX + Keras) and metrics
test_samples/        labelled test images for a quick check
predict.py           command-line inference (ONNX or Keras)
web/                 website: React, TypeScript, Tailwind CSS, ONNX Runtime Web (deployed on Vercel)
dataset/             SOURCES.csv (origin and licence of every image); the images are in the ZIP
results/             metrics, training logs and figures
docs/                preprocessing notes, experiment log
tests/               Python unit tests (pytest); the website has unit (Vitest) and browser tests (Playwright)
```

## Run it yourself

```bash
python -m venv .venv
.venv\Scripts\activate                 # Windows; on Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

# Train on the published dataset: unzip fabricscan-dataset.zip, then
python scripts/train_models.py --splits path/to/fabricscan-dataset --batch-size 16
```

To rebuild the dataset from the original Kaggle downloads instead, place each archive in
`data/external/kaggle/<owner>__<dataset>/archive.zip` and run `import_sources.py`, `clean_dataset.py`,
`split_dataset.py` and `dataset_report.py` before training (the hand-picked images from nexuswho/fabric-defects-dataset
are listed in `dataset/SOURCES.csv`).

Website:

```bash
cd web
npm install
npm run dev          # http://localhost:5173
```

## Licence

Code: [MIT](LICENSE). Images keep the licence of their source dataset (CC BY 4.0, CC BY-NC 4.0 or CC0), listed per
image in `dataset/SOURCES.csv`; please credit the dataset authors above.
