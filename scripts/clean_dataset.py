"""Validate, clean and standardise the raw phone photos (pipeline step 1).

Examples
--------
    python scripts/clean_dataset.py                    # dataset/ -> data/processed/
    python scripts/clean_dataset.py --drop-flagged     # also drop blurry / dark / over-exposed images
    python scripts/clean_dataset.py --input field_test --output data/field_test_processed --reports data/reports/field_test

Outputs: cleaned 256x256 JPEGs in --output, plus clean_manifest.csv, clean_summary.json and a flagged/
folder (images to eyeball) in --reports. See docs/PREPROCESSING.md for every step and threshold.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fabricscan import config  # noqa: E402
from fabricscan.cleaning import CleanConfig, clean_dataset, format_summary  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    defaults = CleanConfig(Path(), Path(), Path())
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--input", type=Path, default=config.DATASET_DIR, help="raw photos: <input>/<class>/...")
    p.add_argument("--output", type=Path, default=config.PROCESSED_DIR, help="cleaned images (rebuilt each run)")
    p.add_argument("--reports", type=Path, default=config.REPORTS_DIR, help="manifest, summary and flagged/")
    p.add_argument("--size", type=int, default=defaults.out_size, help="output side length in pixels")
    p.add_argument("--min-side", type=int, default=defaults.min_side, help="reject images smaller than this")
    p.add_argument("--blur-threshold", type=float, default=defaults.blur_threshold)
    p.add_argument("--dark-threshold", type=float, default=defaults.dark_threshold)
    p.add_argument("--overexposed-fraction", type=float, default=defaults.overexposed_fraction)
    p.add_argument("--duplicate-rmse", type=float, default=defaults.duplicate_rmse,
                   help="thumbnail RMSE at or below which images of different samples count as near duplicates")
    p.add_argument("--burst-rmse", type=float, default=defaults.burst_rmse,
                   help="stricter limit used inside one sample (rotated / re-lit shots are kept)")
    p.add_argument("--time-gap", type=float, default=defaults.time_gap_s,
                   help="flat folders: seconds between shots that still count as one sample")
    p.add_argument("--drop-flagged", action="store_true", help="remove quality-flagged images instead of keeping them")
    p.add_argument("--workers", type=int, default=0, help="worker processes (0 = auto, 1 = no multiprocessing)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = CleanConfig(
        input_dir=args.input,
        output_dir=args.output,
        reports_dir=args.reports,
        out_size=args.size,
        min_side=args.min_side,
        blur_threshold=args.blur_threshold,
        dark_threshold=args.dark_threshold,
        overexposed_fraction=args.overexposed_fraction,
        duplicate_rmse=args.duplicate_rmse,
        burst_rmse=args.burst_rmse,
        time_gap_s=args.time_gap,
        drop_flagged=args.drop_flagged,
        workers=args.workers,
    )
    start = time.perf_counter()
    summary = clean_dataset(cfg)
    if summary["totals"]["files_found"] == 0:
        print(f"\nNo photos found in {args.input}. Put them in <input>/<class>/<sample>/ "
              "(e.g. dataset/hole/s001/IMG_1023.jpg); see docs/DATA_COLLECTION_GUIDE.md.")
        return 1
    print()
    print(format_summary(summary))
    print(f"\nDone in {time.perf_counter() - start:.1f}s. Cleaned images: {args.output}")
    print(f"Audit trail: {args.reports / 'clean_manifest.csv'}  |  review flagged images in {args.reports / 'flagged'}")
    return 0 if summary["totals"]["kept"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
