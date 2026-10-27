"""Build a DEMO model from code-generated fabric images so the website can be tested before real photos exist.

    python scripts/build_demo_model.py            # roughly 20 minutes on a laptop CPU

Steps: generate synthetic images -> clean -> split -> dataset report -> train both models (short schedule) ->
export ONNX -> publish to web/public with "demo": true. The website then shows a DEMO banner. This model has never
seen a real fabric photo, so its answers on real photos are meaningless. Once your own photos are split, train the
real model with scripts/train_models.py (or the notebooks); that overwrites every demo file in web/public.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fabricscan import config  # noqa: E402
from fabricscan.cleaning import CleanConfig, clean_dataset  # noqa: E402
from fabricscan.splitting import SplitConfig, format_split_summary, split_dataset  # noqa: E402
from fabricscan.synthetic import generate_dataset  # noqa: E402
from fabricscan.workflow import TrainingPlan, run_training  # noqa: E402

DEMO = config.DATA_DIR / "_demo"


def main() -> int:
    print("1/5 Generating synthetic fabric images ...")
    generate_dataset(DEMO / "raw", samples_per_class=25, shots=4, size=384, seed=7, inject_faults=False)
    print("2/5 Cleaning ...")
    clean_dataset(CleanConfig(DEMO / "raw", DEMO / "processed", DEMO / "reports"), log=lambda _: None)
    print("3/5 Splitting ...")
    summary = split_dataset(SplitConfig(DEMO / "processed", DEMO / "splits", DEMO / "reports"), log=lambda _: None)
    print(format_split_summary(summary))
    print("4/5 Dataset report ...")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "dataset_report.py"), "--data", str(DEMO / "splits"),
                    "--reports", str(DEMO / "reports"), "--capture-log", str(DEMO / "none.csv"),
                    "--out", str(DEMO / "results")], check=True, stdout=subprocess.DEVNULL)
    print("5/5 Training both models (short demo schedule) ...")
    # A from-scratch CNN needs many epochs and small batches on 408 images (see docs/EXPERIMENT_LOG.md).
    plan = TrainingPlan(epochs_scratch=50, epochs_frozen=10, epochs_finetune=10, patience_scratch=15,
                        patience_transfer=5, batch_size=16)
    run_training(DEMO / "splits", plan=plan, models_dir=config.MODELS_DIR / "_demo",
                 results_dir=DEMO / "results" / "metrics", dataset_summary=DEMO / "results" / "dataset_summary.json",
                 demo=True)
    print("\nDemo model is ready. Start the website:  cd web ; npm run dev")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
