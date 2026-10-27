"""Save a grid showing what the training augmentation does (figure for the report).

    python scripts/preview_augmentation.py         # uses data/splits/train
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import tensorflow as tf  # noqa: E402

from fabricscan import config  # noqa: E402
from fabricscan.data import augment, list_split, load_image, resize_for_model  # noqa: E402

SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#898781"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--split-dir", type=Path, default=config.SPLITS_DIR / "train")
    p.add_argument("--out", type=Path, default=config.RESULTS_DIR / "dataset" / "augmentation_preview.png")
    p.add_argument("--rows", type=int, default=4, help="number of example images (from different classes)")
    p.add_argument("--variants", type=int, default=6, help="augmented versions per image")
    p.add_argument("--seed", type=int, default=config.SEED)
    args = p.parse_args(argv)

    paths, labels = list_split(args.split_dir)
    rng = random.Random(args.seed)
    by_class: dict[int, list[str]] = {}
    for path, label in zip(paths, labels):
        by_class.setdefault(label, []).append(path)
    chosen = rng.sample(sorted(by_class), min(args.rows, len(by_class)))
    picks = [(rng.choice(by_class[c]), c) for c in chosen]

    plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans", "sans-serif"]
    cols = args.variants + 1
    fig, axes = plt.subplots(len(picks), cols, figsize=(cols * 1.5, len(picks) * 1.62), dpi=150, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for r, (path, label) in enumerate(picks):
        image = load_image(tf.constant(path))
        views = [resize_for_model(image)] + [
            augment(image, tf.constant([args.seed, r * 100 + k], dtype=tf.int64)) for k in range(args.variants)
        ]
        for k, view in enumerate(views):
            ax = axes[r, k]
            ax.imshow(tf.cast(view, tf.uint8).numpy())
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if r == 0:
                ax.set_title("original" if k == 0 else f"augmented {k}", fontsize=8, color=MUTED)
        axes[r, 0].set_ylabel(config.DISPLAY_NAMES[config.CLASS_NAMES[label]], fontsize=8, color=INK)
    fig.suptitle("Training-time augmentation (rotation, zoom, flips, brightness, contrast, noise)", x=0.01,
                 ha="left", fontsize=10, color=INK, fontweight="semibold")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, facecolor=SURFACE, bbox_inches="tight")
    print(f"Saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
