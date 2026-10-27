"""Field test: does the model call clean, worn garments "good"? (The live-camera situation.)

    python scripts/field_test.py            # writes results/metrics/field_test.json

Uses photos from Kaggle rajkumarl/people-clothing-segmentation that are NOT in the dataset (not in dataset/SOURCES.csv):
a square crop inside the largest upper-body garment, exactly like the training crops, then the website model.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import import_sources  # noqa: E402  (scripts/import_sources.py)
from fabricscan import config  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", type=Path, default=config.PROJECT_ROOT / "weights" / "fabricscan.onnx")
    p.add_argument("--count", type=int, default=150)
    p.add_argument("--out", type=Path, default=config.RESULTS_DIR / "metrics" / "field_test.json")
    args = p.parse_args(argv)

    import onnxruntime as ort

    source = import_sources.MASK_SOURCES[0]
    root = import_sources.extract(source.ref)
    with open(config.DATASET_DIR / "SOURCES.csv", encoding="utf-8") as fh:
        used = {r["original"] for r in csv.DictReader(fh) if source.ref in r["source"]}
    pairs = [(i, m) for i, m in zip(sorted((root / source.images).glob("*.png")), sorted((root / source.masks).glob("*.png")))
             if i.relative_to(root).as_posix() not in used]
    random.Random(7).shuffle(pairs)

    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    classes = json.loads(args.model.with_name("model_meta.json").read_text(encoding="utf-8"))["class_names"]
    predictions = []
    for image_path, mask_path in pairs:
        if len(predictions) >= args.count:
            break
        with Image.open(mask_path) as m:
            garment = np.isin(np.asarray(m), source.garment_ids)
        ys, xs = np.nonzero(garment)
        if len(xs) < 0.06 * garment.size:
            continue
        x0, x1, y0, y1 = np.percentile(xs, 10), np.percentile(xs, 90), np.percentile(ys, 10), np.percentile(ys, 90)
        with Image.open(image_path) as im:
            im = im.convert("RGB")
            w, h = im.size
            side = int(min(w, h, max(160, min(x1 - x0, y1 - y0) * 0.85)))
            x = int(max(0, min(w - side, (x0 + x1) / 2 - side / 2)))
            y = int(max(0, min(h - side, (y0 + y1) / 2 - side / 2)))
            crop = im.crop((x, y, x + side, y + side)).resize((256, 256), Image.Resampling.BILINEAR)
        batch = np.asarray(crop.resize((224, 224), Image.Resampling.BILINEAR), np.float32)[None]
        probs = session.run(["probs"], {session.get_inputs()[0].name: batch})[0][0]
        predictions.append(classes[int(np.argmax(probs))])

    counts = Counter(predictions)
    result = {"description": "clean worn garments not in the dataset (people-clothing-segmentation), garment crop",
              "images": len(predictions), "predicted": dict(counts),
              "good_rate": round(counts["good"] / max(len(predictions), 1), 4)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"{len(predictions)} unseen worn clean garments: {dict(counts)} -> {result['good_rate']:.1%} called good")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
