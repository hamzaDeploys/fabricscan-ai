"""End-to-end smoke test of the pipeline on procedurally generated images (no real photos needed).

    python scripts/smoke_test.py

Runs every pipeline script exactly as you would on the real dataset, but on synthetic data in
data/_smoke_test/, then verifies the results (planted faults caught, leakage-free split, figures written,
tf.data pipeline output). Nothing here touches dataset/, data/processed, data/splits or results/.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # hide TensorFlow's start-up info/warning chatter
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fabricscan import config  # noqa: E402

SMOKE = config.SMOKE_DIR
RAW, PROCESSED, SPLITS = SMOKE / "raw", SMOKE / "processed", SMOKE / "splits"
REPORTS, RESULTS = SMOKE / "reports", SMOKE / "results"

checks: list[tuple[str, str, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    checks.append(("PASS" if ok else "FAIL", name, detail))


def run(script: str, *args: str) -> None:
    cmd = [sys.executable, str(ROOT / "scripts" / script), *map(str, args)]
    print(f"\n$ python scripts/{script} {' '.join(map(str, args))}")
    start = time.perf_counter()
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    print(proc.stdout.rstrip())
    if proc.returncode != 0:
        print(proc.stderr.rstrip())
    check(f"{script} exits cleanly", proc.returncode == 0, f"{time.perf_counter() - start:.1f}s")
    if proc.returncode != 0:
        raise SystemExit(report())


def verify_cleaning(expected: dict[str, str]) -> None:
    with open(REPORTS / "clean_manifest.csv", newline="", encoding="utf-8") as fh:
        manifest = {row["source"]: row for row in csv.DictReader(fh)}
    for rel, outcome in sorted(expected.items()):
        row = manifest.get(rel)
        if outcome == "ignored":
            check(f"ignored: {rel}", row is None)
        elif row is None:
            check(f"{outcome}: {rel}", False, "missing from manifest")
        elif outcome == "kept":
            check(f"kept: {rel}", row["status"] == "kept", row["reason"])
        elif outcome.startswith("flag:"):
            flag = outcome[5:]
            check(f"flagged {flag}: {rel}", row["status"] == "kept" and flag in row["flags"].split(";"),
                  row["flags"] or row["reason"])
        else:
            check(f"removed as {outcome}: {rel}", row["status"] == "removed" and row["reason"].startswith(outcome),
                  row["reason"])
    planted = sum(1 for o in expected.values() if o == "near_duplicate_of")
    removed = [r["source"] for r in manifest.values() if r["reason"].startswith("near_duplicate_of")]
    check("no false near-duplicate removals (rotated / re-lit shots kept)", len(removed) == planted,
          f"{len(removed)} removed, {planted} planted")
    flat = {row["group"] for row in manifest.values() if row["class"] == "broken_thread" and row["status"] == "kept"}
    check("flat folder grouped by capture time", len(flat) == 12, f"{len(flat)} groups for 12 samples")
    sizes = set()
    for path in list(PROCESSED.rglob("*.jpg"))[:40]:
        from PIL import Image

        with Image.open(path) as img:
            sizes.add((img.size, img.mode, bool(img.getexif())))
    check("outputs are 256x256 RGB without EXIF", sizes == {((256, 256), "RGB", False)}, str(sizes))


def verify_split() -> None:
    summary = json.loads((REPORTS / "split_summary.json").read_text(encoding="utf-8"))
    audit = summary["audit"]
    check("no sample group shared between splits", not audit["shared_groups"], str(audit["shared_groups"][:3]))
    check("no near-duplicates across splits", not audit["cross_split_near_duplicates"],
          str(audit["cross_split_near_duplicates"][:3]))
    off = {s: abs(summary["achieved_ratios"][s] - r) for s, r in config.SPLIT_RATIOS.items()}
    check("split ratios within 6 points of 70/15/15", max(off.values()) <= 0.06, str(summary["achieved_ratios"]))
    every = all(summary["images"][s][c] > 0 for s in config.SPLITS for c in config.CLASS_NAMES)
    check("every class present in every split", every)


def verify_tf_pipeline() -> None:
    import numpy as np
    import tensorflow as tf

    from fabricscan.data import class_weights, list_split, make_dataset

    train = make_dataset(SPLITS / "train", training=True, batch_size=8)
    images, labels = next(iter(train))
    check("train batch shape (8, 224, 224, 3) float32", tuple(images.shape) == (8, 224, 224, 3)
          and images.dtype == tf.float32, str(images.shape))
    lo, hi = float(tf.reduce_min(images)), float(tf.reduce_max(images))
    check("pixel range stays within [0, 255]", 0.0 <= lo and hi <= 255.0, f"{lo:.1f}..{hi:.1f}")
    val = make_dataset(SPLITS / "val", training=False, batch_size=8)
    first = next(iter(val))[0].numpy()
    again = next(iter(val))[0].numpy()
    check("validation pipeline is deterministic", np.array_equal(first, again))
    _, y = list_split(SPLITS / "train")
    weights = class_weights(y)
    check(f"class weights for all {config.NUM_CLASSES} classes", len(weights) == config.NUM_CLASSES, json.dumps({k: round(v, 2) for k, v in weights.items()}))


def report() -> int:
    width = max(len(name) for _, name, _ in checks) + 2
    print("\n" + "=" * 78 + "\nSMOKE TEST RESULTS\n" + "=" * 78)
    for status, name, detail in checks:
        print(f"[{status}] {name:<{width}} {detail}")
    failed = sum(1 for s, _, _ in checks if s == "FAIL")
    print("-" * 78)
    print(f"{len(checks) - failed} passed, {failed} failed")
    return 1 if failed else 0


def main() -> int:
    from fabricscan.synthetic import generate_dataset

    print(f"Generating synthetic raw dataset in {RAW} ...")
    result = generate_dataset(RAW, samples_per_class=12, shots=4, size=512, seed=0)
    check("synthetic dataset generated", sum(result["images"].values()) == config.NUM_CLASSES * 12 * 4)

    run("clean_dataset.py", "--input", RAW, "--output", PROCESSED, "--reports", REPORTS)
    verify_cleaning(result["expected"])
    run("split_dataset.py", "--input", PROCESSED, "--output", SPLITS, "--reports", REPORTS)
    verify_split()
    run("dataset_report.py", "--data", SPLITS, "--reports", REPORTS, "--capture-log", SMOKE / "no_log.csv",
        "--out", RESULTS)
    for name in ("class_distribution.png", "sample_grid.png", "dataset_summary.md", "dataset_summary.json"):
        check(f"report wrote {name}", (RESULTS / name).is_file())

    if importlib.util.find_spec("tensorflow") is None:
        checks.append(("SKIP", "TensorFlow pipeline checks", "tensorflow not installed"))
    else:
        try:
            import tensorflow  # noqa: F401
        except ImportError as exc:
            checks.append(("SKIP", "TensorFlow pipeline checks", f"tensorflow cannot load: {exc}"[:120]))
        else:
            run("preview_augmentation.py", "--split-dir", SPLITS / "train", "--out", RESULTS / "augmentation_preview.png")
            verify_tf_pipeline()
    return report()


if __name__ == "__main__":
    raise SystemExit(main())
