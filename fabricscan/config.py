"""Project-wide constants shared by the data scripts, notebooks, tests and (via model_meta.json) the web app.

This module must stay importable without TensorFlow so the data scripts run on any machine.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Order matters: index i is the model's output unit i. Never reorder after training.
CLASS_NAMES: tuple[str, ...] = (
    "good",
    "hole",
    "stain",
    "broken_thread",
)
DISPLAY_NAMES: dict[str, str] = {
    "good": "Good (No Defect)",
    "hole": "Hole",
    "stain": "Stain / Oil Spot",
    "broken_thread": "Broken Thread / Slub",
}
NUM_CLASSES = len(CLASS_NAMES)

SEED = 42
PROCESSED_SIZE = 256  # side length of the cleaned square images in data/processed
IMG_SIZE = 224  # model input side length
SPLIT_RATIOS: dict[str, float] = {"train": 0.70, "val": 0.15, "test": 0.15}
SPLITS: tuple[str, ...] = tuple(SPLIT_RATIOS)

IMAGE_EXTENSIONS = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".heic", ".heif"}
)

# Processed file names are "<group>__<original stem>.jpg"; the group is the physical sample.
GROUP_SEPARATOR = "__"

# Folders created by the pipeline carry this marker so they can be safely wiped and rebuilt.
GENERATED_MARKER = ".fabricscan_generated"

# Default locations (every script lets you override them on the command line)
DATASET_DIR = PROJECT_ROOT / "dataset"
CAPTURE_LOG = DATASET_DIR / "capture_log.csv"
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"
REPORTS_DIR = DATA_DIR / "reports"
SMOKE_DIR = DATA_DIR / "_smoke_test"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"

# Collection targets from docs/DATA_COLLECTION_GUIDE.md
MIN_IMAGES_PER_CLASS = 150
TARGET_IMAGES_PER_CLASS = 250
MAX_IMBALANCE_RATIO = 1.5
