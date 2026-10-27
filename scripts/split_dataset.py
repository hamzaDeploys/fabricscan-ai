"""Group-aware, stratified, seeded 70/15/15 train/val/test split (pipeline step 2).

Examples
--------
    python scripts/split_dataset.py                    # data/processed -> data/splits/{train,val,test}/<class>/
    python scripts/split_dataset.py --zip              # also write data/fabricscan_splits.zip (for Google Colab)
    python scripts/split_dataset.py --seed 7 --ratios 0.7 0.15 0.15
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fabricscan import config  # noqa: E402
from fabricscan.splitting import SplitConfig, format_split_summary, split_dataset  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--input", type=Path, default=config.PROCESSED_DIR, help="cleaned images: <input>/<class>/*.jpg")
    p.add_argument("--output", type=Path, default=config.SPLITS_DIR, help="split folders (rebuilt each run)")
    p.add_argument("--reports", type=Path, default=config.REPORTS_DIR, help="split_manifest.csv + split_summary.json")
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--ratios", type=float, nargs=3, default=list(config.SPLIT_RATIOS.values()),
                   metavar=("TRAIN", "VAL", "TEST"))
    p.add_argument("--no-audit", action="store_true", help="skip the cross-split near-duplicate check")
    p.add_argument("--zip", action="store_true", help="also zip the splits for uploading to Google Drive / Colab")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = SplitConfig(
        input_dir=args.input,
        output_dir=args.output,
        reports_dir=args.reports,
        ratios=dict(zip(config.SPLITS, args.ratios)),
        seed=args.seed,
        audit=not args.no_audit,
    )
    summary = split_dataset(cfg)
    print()
    print(format_split_summary(summary))
    if args.zip:
        archive = shutil.make_archive(str(args.output.parent / "fabricscan_splits"), "zip", args.output.parent,
                                      args.output.name)
        print(f"\nZipped for Colab: {archive}")
    audit = summary.get("audit") or {}
    return 1 if audit.get("shared_groups") else 0


if __name__ == "__main__":
    raise SystemExit(main())
