"""Small filesystem helpers shared by the pipeline steps."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

from .config import GENERATED_MARKER


def prepare_output_dir(path: Path) -> Path:
    """Create an empty output folder, wiping it first only if this pipeline created it.

    Refuses to delete a non-empty folder without the marker file, so a mistyped --output can never erase
    the raw photos or anything else.
    """
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        if not (path / GENERATED_MARKER).exists():
            raise RuntimeError(
                f"'{path}' is not empty and was not created by the FabricScan scripts, so it will not be "
                "deleted. Choose another output folder or empty it yourself."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    (path / GENERATED_MARKER).write_text(
        "Created by the FabricScan AI scripts. Safe to delete: it is rebuilt on every run.\n",
        encoding="utf-8",
    )
    return path


def safe_name(name: str, fallback: str = "x") -> str:
    """File-name-safe token: letters, digits, '-' and single '_' only (never the '__' group separator)."""
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "-", name)
    cleaned = re.sub(r"_{2,}", "_", cleaned).strip("-_")
    return cleaned or fallback
