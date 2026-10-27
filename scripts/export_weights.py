"""Publish the trained models in weights/ (committed to GitHub) and a small labelled test set in test_samples/.

    python scripts/export_weights.py           # run automatically at the end of scripts/train_models.py

weights/
    fabricscan.onnx        the best model (the one the website runs), ONNX opset 17
    scratch_cnn.keras / .onnx, mobilenetv2.keras / .onnx    both trained models
    model_meta.json        class order, input format, chosen model, test accuracy
    metrics.json           full test metrics and training history of both models
test_samples/<class>/      a few images per class from the test split (never seen in training)

Check them with:  python predict.py test_samples --eval
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fabricscan import config  # noqa: E402
from fabricscan.export import write_model_meta  # noqa: E402
from fabricscan.workflow import MODEL_INFO  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--models", type=Path, default=config.MODELS_DIR)
    p.add_argument("--metrics", type=Path, default=config.RESULTS_DIR / "metrics" / "metrics.json")
    p.add_argument("--splits", type=Path, default=config.SPLITS_DIR)
    p.add_argument("--weights", type=Path, default=config.PROJECT_ROOT / "weights")
    p.add_argument("--samples", type=Path, default=config.PROJECT_ROOT / "test_samples")
    p.add_argument("--per-class", type=int, default=10)
    args = p.parse_args(argv)
    if not args.metrics.is_file():
        print(f"No {args.metrics}; train first with scripts/train_models.py")
        return 1
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    if metrics.get("demo"):
        print("These are demo-model metrics; weights/ is only for models trained on the real dataset.")
        return 1

    args.weights.mkdir(parents=True, exist_ok=True)
    for key in metrics["models"]:
        for ext in (".keras", ".onnx"):
            src = args.models / f"{MODEL_INFO[key]['file']}{ext}"
            if src.is_file():
                shutil.copyfile(src, args.weights / src.name)
    best = metrics["best_model"]
    entry = metrics["models"][best]
    shutil.copyfile(args.models / f"{MODEL_INFO[best]['file']}.onnx", args.weights / "fabricscan.onnx")
    write_model_meta(args.weights / "model_meta.json", model_key=best, architecture=entry["name"],
                     cam_shape=entry["onnx_check"]["cam_shape"], demo=False,
                     test_accuracy=entry["test"]["accuracy"], created=metrics["generated"])
    shutil.copyfile(args.metrics, args.weights / "metrics.json")

    if args.samples.exists():
        shutil.rmtree(args.samples)
    rng = random.Random(config.SEED)
    for cls in config.CLASS_NAMES:
        files = sorted((args.splits / "test" / cls).glob("*.jpg"))
        (args.samples / cls).mkdir(parents=True)
        for f in rng.sample(files, min(args.per_class, len(files))):
            shutil.copyfile(f, args.samples / cls / f.name)

    sizes = ", ".join(f"{f.name} {f.stat().st_size / 1e6:.1f} MB" for f in sorted(args.weights.glob("*.*")))
    print(f"weights/: {sizes}")
    print(f"test_samples/: {args.per_class} test images per class. Check: python predict.py test_samples --eval")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
