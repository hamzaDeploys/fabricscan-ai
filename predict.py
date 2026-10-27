"""FabricScan AI: classify fabric photos with the trained model (good, hole, stain, broken_thread).

    pip install numpy pillow onnxruntime
    python predict.py photo.jpg                      # one image
    python predict.py photo1.jpg photos/             # several images and/or folders
    python predict.py test_samples --eval            # folder with one sub-folder per class: accuracy + confusion matrix
    python predict.py photo.jpg --model weights/mobilenetv2.keras   # a Keras model instead (needs tensorflow)

The default model is weights/fabricscan.onnx (the best of the two trained models). Preprocessing matches training:
EXIF rotation, centre square crop, resize to 256 then 224 px, raw RGB values 0-255 (normalisation is inside the model).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "weights" / "fabricscan.onnx"
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
FALLBACK_CLASSES = ["good", "hole", "stain", "broken_thread"]


def load_classes(model_path: Path) -> list[str]:
    meta = model_path.with_name("model_meta.json")
    if meta.is_file():
        return json.loads(meta.read_text(encoding="utf-8"))["class_names"]
    return FALLBACK_CLASSES


def preprocess(path: Path, size: int = 224) -> np.ndarray:
    """(size, size, 3) float32 RGB in 0..255, prepared exactly like the training images."""
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        side = min(img.size)
        left, top = (img.width - side) // 2, (img.height - side) // 2
        img = img.crop((left, top, left + side, top + side))
        img = img.resize((256, 256), Image.Resampling.BILINEAR).resize((size, size), Image.Resampling.BILINEAR)
        return np.asarray(img, dtype=np.float32)


class Model:
    def __init__(self, path: Path):
        self.path = path
        if path.suffix == ".onnx":
            import onnxruntime as ort

            self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            self.input = self.session.get_inputs()[0].name
            self.size = self.session.get_inputs()[0].shape[1] or 224
        else:
            import keras

            self.keras = keras.models.load_model(path, compile=False)
            self.size = self.keras.input_shape[1] or 224

    def predict(self, batch: np.ndarray) -> np.ndarray:
        if hasattr(self, "session"):
            return self.session.run(["probs"], {self.input: batch})[0]
        return np.asarray(self.keras.predict(batch, verbose=0))


def collect(paths: list[str]) -> list[Path]:
    files = []
    for p in map(Path, paths):
        if p.is_dir():
            files += sorted(f for f in p.rglob("*") if f.suffix.lower() in IMAGE_SUFFIXES)
        elif p.is_file():
            files.append(p)
        else:
            print(f"Not found: {p}", file=sys.stderr)
    return files


def evaluate(rows: list[dict], classes: list[str]) -> None:
    """Accuracy, per-class precision/recall/F1 and the confusion matrix, using each file's parent folder as the label."""
    labelled = [r for r in rows if r["true"] in classes]
    if not labelled:
        print("\n--eval: no image sits in a folder named after a class " + str(classes))
        return
    k = len(classes)
    cm = np.zeros((k, k), dtype=int)
    for r in labelled:
        cm[classes.index(r["true"]), classes.index(r["predicted"])] += 1
    acc = np.trace(cm) / cm.sum()
    print(f"\nAccuracy: {acc:.4f} ({np.trace(cm)}/{cm.sum()} images)\n")
    print(f"{'class':<16}{'precision':>10}{'recall':>9}{'f1':>8}{'images':>8}")
    f1s = []
    for i, c in enumerate(classes):
        tp, pred, true = cm[i, i], cm[:, i].sum(), cm[i, :].sum()
        p = tp / pred if pred else 0.0
        r = tp / true if true else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        f1s.append(f1)
        print(f"{c:<16}{p:>10.3f}{r:>9.3f}{f1:>8.3f}{true:>8d}")
    print(f"{'macro F1':<16}{np.mean(f1s):>27.3f}\n")
    print("Confusion matrix (rows = true class, columns = predicted):")
    print(" " * 16 + "".join(f"{c[:8]:>9}" for c in classes))
    for i, c in enumerate(classes):
        print(f"{c:<16}" + "".join(f"{v:>9d}" for v in cm[i]))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("inputs", nargs="+", help="image files and/or folders")
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="weights/*.onnx (default) or weights/*.keras")
    p.add_argument("--eval", action="store_true", help="use each image's folder name as its true class and score the model")
    p.add_argument("--csv", type=Path, help="also write the predictions to this CSV file")
    p.add_argument("--batch-size", type=int, default=16)
    args = p.parse_args(argv)

    if not args.model.is_file():
        print(f"Model not found: {args.model}", file=sys.stderr)
        return 1
    classes = load_classes(args.model)
    model = Model(args.model)
    files = collect(args.inputs)
    if not files:
        print("No images found.", file=sys.stderr)
        return 1

    rows = []
    for start in range(0, len(files), args.batch_size):
        chunk = files[start:start + args.batch_size]
        probs = model.predict(np.stack([preprocess(f, model.size) for f in chunk]))
        for f, pr in zip(chunk, probs):
            best = int(np.argmax(pr))
            rows.append({"file": str(f), "predicted": classes[best], "confidence": round(float(pr[best]), 4),
                         "true": f.parent.name, **{f"p_{c}": round(float(v), 4) for c, v in zip(classes, pr)}})
            print(f"{classes[best]:<14} {pr[best]:6.1%}  {f}")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nPredictions written to {args.csv}")
    if args.eval:
        evaluate(rows, classes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
