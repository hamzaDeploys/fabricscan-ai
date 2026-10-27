"""Write a procedurally generated, raw-format dataset for smoke-testing the pipeline.

These images are drawn by code (no photos, no public datasets). They go to data/_smoke_test/raw by default and
must NEVER be copied into dataset/ or used to train the submitted model.

    python scripts/make_synthetic_dataset.py
    python scripts/make_synthetic_dataset.py --samples 20 --shots 4 --size 512
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fabricscan import config  # noqa: E402
from fabricscan.synthetic import generate_dataset  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--output", type=Path, default=config.SMOKE_DIR / "raw")
    p.add_argument("--samples", type=int, default=12, help="physical samples per class")
    p.add_argument("--shots", type=int, default=4, help="shots per sample")
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-faults", action="store_true", help="do not plant corrupt/duplicate/... test files")
    args = p.parse_args(argv)
    result = generate_dataset(args.output, args.samples, args.shots, args.size, args.seed,
                              inject_faults=not args.no_faults)
    print(f"Wrote {sum(result['images'].values())} synthetic images + {len(result['expected'])} planted test "
          f"files to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
