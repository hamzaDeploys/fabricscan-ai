"""Train both models on the real dataset, evaluate, export to ONNX and publish the best one to the website.

    python scripts/train_models.py                       # uses data/splits, trains both models
    python scripts/train_models.py --models transfer     # only MobileNetV2
    python scripts/train_models.py --no-web              # do not touch web/public

Outputs: models/*.keras + models/*.onnx, results/metrics/metrics.json (+ training logs), and, unless --no-web,
web/public/models/fabricscan.onnx, model_meta.json, data/metrics.json and samples/. The notebooks in notebooks/
run the same steps cell by cell with plots.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import export_weights  # noqa: E402  (scripts/export_weights.py)
import pick_samples  # noqa: E402  (scripts/pick_samples.py)
from fabricscan import config  # noqa: E402
from fabricscan.workflow import TrainingPlan, run_training  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    defaults = TrainingPlan()
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--splits", type=Path, default=config.SPLITS_DIR)
    p.add_argument("--models", nargs="+", choices=["scratch", "transfer"], default=["scratch", "transfer"])
    p.add_argument("--epochs-scratch", type=int, default=defaults.epochs_scratch)
    p.add_argument("--epochs-frozen", type=int, default=defaults.epochs_frozen)
    p.add_argument("--epochs-finetune", type=int, default=defaults.epochs_finetune)
    p.add_argument("--batch-size", type=int, default=defaults.batch_size)
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--no-web", action="store_true", help="do not copy the best model into web/public")
    args = p.parse_args(argv)
    if not (args.splits / "train").is_dir():
        print(f"No split found in {args.splits}. Run scripts/clean_dataset.py and scripts/split_dataset.py first.")
        return 1
    plan = TrainingPlan(epochs_scratch=args.epochs_scratch, epochs_frozen=args.epochs_frozen,
                        epochs_finetune=args.epochs_finetune, batch_size=args.batch_size, seed=args.seed)
    run_training(args.splits, models=args.models, plan=plan,
                 web_public=None if args.no_web else config.PROJECT_ROOT / "web" / "public")
    if not args.no_web and pick_samples.main([]) != 0:
        return 1
    return export_weights.main([])


if __name__ == "__main__":
    raise SystemExit(main())
