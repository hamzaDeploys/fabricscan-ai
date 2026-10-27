"""Choose the website's sample images: clear test-split examples, from different sources, that the model gets right.

    python scripts/pick_samples.py           # run automatically at the end of scripts/train_models.py

Images listed in assets/held_out_samples/samples.json come first: photos deliberately kept out of the dataset (see
HELD_OUT in import_sources.py), used if the model gets them right. The rest come from the test split: for every class
the website model (web/public/models/fabricscan.onnx) keeps the images it classifies correctly with high confidence
and takes the most confident one from each preferred source. None of them was used in training. The honest overall
numbers stay in the Performance section (results/metrics/metrics.json).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fabricscan import config  # noqa: E402
from fabricscan.splitting import group_of, group_sources  # noqa: E402

# Sources to show first, per class (substrings of the source label), so the strip shows garments as well as fabric.
PREFERRED = {
    "good": ["people-clothing", "clothes-staindefect", "fabric-defects-dataset", "stitchingnet"],
    "hole": ["fabric-defect-without", "fabric-defects-dataset"],
    "stain": ["clothes-staindefect", "fabric-defects-dataset", "fabric-defect-without"],
    "broken_thread": ["stitchingnet", "fabric-defect-without", "fabric-defects-dataset"],
}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--test", type=Path, default=config.SPLITS_DIR / "test")
    p.add_argument("--web", type=Path, default=config.PROJECT_ROOT / "web" / "public")
    p.add_argument("--per-class", type=int, default=2)
    p.add_argument("--min-confidence", type=float, default=0.9)
    args = p.parse_args(argv)

    import onnxruntime as ort

    meta = json.loads((args.web / "models" / "model_meta.json").read_text(encoding="utf-8"))
    classes, size = meta["class_names"], meta["input"]["size"]
    session = ort.InferenceSession(str(args.web / "models" / "fabricscan.onnx"), providers=["CPUExecutionProvider"])
    sources = group_sources(config.DATASET_DIR / "SOURCES.csv")
    held_dir = config.PROJECT_ROOT / "assets" / "held_out_samples"
    held = json.loads((held_dir / "samples.json").read_text(encoding="utf-8")) if (held_dir / "samples.json").is_file() else []

    def predict(path: Path) -> np.ndarray:
        with Image.open(path) as im:
            x = np.asarray(im.convert("RGB").resize((size, size), Image.Resampling.BILINEAR), np.float32)[None]
        return session.run(["probs"], {session.get_inputs()[0].name: x})[0][0]

    samples_dir = args.web / "samples"
    if samples_dir.exists():
        shutil.rmtree(samples_dir)
    samples_dir.mkdir(parents=True)
    manifest = []
    for cls in classes:
        picks = []
        for h in (h for h in held if h["class"] == cls):
            probs = predict(held_dir / h["file"])
            # Hand-picked photos only need to be right and above the website's low-confidence line (60 %).
            ok = classes[int(np.argmax(probs))] == cls and probs.max() >= 0.6
            print(f"held-out {h['file']}: {classes[int(np.argmax(probs))]} {probs.max():.1%}{'' if ok else ' (not used)'}")
            if ok and len(picks) < args.per_class:
                picks.append((f"{h['source']}, held out of the dataset", (float(probs.max()), held_dir / h["file"])))
        by_source: dict[str, list[tuple[float, Path]]] = defaultdict(list)
        for f in sorted((args.test / cls).glob("*.jpg")):
            probs = predict(f)
            if classes[int(np.argmax(probs))] == cls and probs.max() >= args.min_confidence:
                by_source[sources.get(group_of(f), "Own photo")].append((float(probs.max()), f))
        order = sorted(by_source, key=lambda s: next((i for i, k in enumerate(PREFERRED.get(cls, [])) if k in s), 99))
        picks += [(f"{s}, test split", max(by_source[s])) for s in order][: args.per_class - len(picks)]
        if len(picks) < args.per_class:  # not enough sources: take the next most confident images
            taken = {f for _, (_, f) in picks}
            rest = sorted((c for s in order for c in by_source[s] if c[1] not in taken), reverse=True)
            picks += [(f"{sources.get(group_of(f), 'Own photo')}, test split", (c, f)) for c, f in rest[: args.per_class - len(picks)]]
        for i, (source, (conf, f)) in enumerate(picks, 1):
            name = f"{cls}-{i}.jpg"
            shutil.copyfile(f, samples_dir / name)
            manifest.append({"file": f"samples/{name}", "class": cls, "label": meta["display_names"][cls],
                             "source": source})
            print(f"{name:<22} {conf:6.1%}  {source}")
    (samples_dir / "samples.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
