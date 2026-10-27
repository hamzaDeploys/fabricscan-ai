"""Step 2 of the data pipeline: a group-aware, stratified, seeded train/val/test split.

* **Stratified:** every class is split separately, so each split keeps the class proportions.
* **Group-aware:** all shots of one physical sample (the "<group>__" prefix written by the cleaning step)
  go to the same split. Otherwise near-identical shots of the same defect would sit in both train and
  test, and the test accuracy would be inflated (data leakage).
* **Seeded:** the same seed and the same input always give the same split.

Groups are assigned largest-first (random order among equal sizes) to whichever split is furthest below
its target share, which lands within a few images of the requested 70/15/15 ratio.
After copying, a leakage audit re-checks that no group and no near-duplicate image crosses splits.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from PIL import Image

from .config import CLASS_NAMES, GROUP_SEPARATOR, SEED, SPLIT_RATIOS
from .fsutils import prepare_output_dir, safe_name
from .imaging import rmse_matrix, thumbnail

_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


@dataclass
class SplitConfig:
    input_dir: Path
    output_dir: Path
    reports_dir: Path
    ratios: dict[str, float] = field(default_factory=lambda: dict(SPLIT_RATIOS))
    seed: int = SEED
    class_names: Sequence[str] = CLASS_NAMES
    audit: bool = True
    duplicate_rmse: float = 3.0  # same meaning as in the cleaning step


def group_of(path: Path) -> str:
    """Physical-sample group encoded in a processed file name ('<group>__<stem>.jpg')."""
    stem = Path(path).stem
    return stem.split(GROUP_SEPARATOR, 1)[0] if GROUP_SEPARATOR in stem else stem


def group_sources(sources_csv: Path) -> dict[str, str]:
    """{group: "source (licence)"} from dataset/SOURCES.csv; groups missing from it are the author's own photos."""
    table: dict[str, str] = {}
    if Path(sources_csv).is_file():
        with open(sources_csv, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                parts = row["file"].split("/")
                if len(parts) >= 3:
                    licence = (row.get("licence") or "").strip()
                    table[safe_name(parts[1], "sample")] = row["source"] + (f" ({licence})" if licence else "")
    return table


def assign_groups_to_splits(
    group_sizes: dict[str, int], ratios: dict[str, float], rng: random.Random
) -> dict[str, str]:
    """Map each group to a split so that image counts approach the requested ratios."""
    if len(group_sizes) < len(ratios):
        raise ValueError(f"Need at least {len(ratios)} groups, got {len(group_sizes)}.")
    names = sorted(group_sizes)
    rng.shuffle(names)
    names.sort(key=lambda g: -group_sizes[g])  # stable sort: random order among equal sizes
    total = sum(group_sizes.values())
    targets = {s: r * total for s, r in ratios.items()}
    counts = dict.fromkeys(ratios, 0)
    assignment: dict[str, str] = {}
    for g in names:
        split = max(ratios, key=lambda s: (targets[s] - counts[s]) / targets[s])
        assignment[g] = split
        counts[split] += group_sizes[g]
    return assignment


def _validate_ratios(ratios: dict[str, float]) -> None:
    if any(r <= 0 for r in ratios.values()) or abs(sum(ratios.values()) - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must be positive and sum to 1, got {ratios}.")


def _audit(rows: list[dict], output_dir: Path, cfg: SplitConfig) -> dict:
    """Re-check the written split for leakage: shared groups, and near-duplicate images across splits."""
    groups_by_split: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        groups_by_split[(row["class"], row["split"])].add(row["group"])
    shared_groups = []
    for class_name in cfg.class_names:
        splits = list(cfg.ratios)
        for i, a in enumerate(splits):
            for b in splits[i + 1:]:
                for g in sorted(groups_by_split[(class_name, a)] & groups_by_split[(class_name, b)]):
                    shared_groups.append(f"{class_name}/{g} in {a} and {b}")

    thumbs = []
    for row in rows:
        with Image.open(output_dir / row["path"]) as img:
            thumbs.append(thumbnail(img))
    near_dupes = []
    if len(rows) > 1:
        distances = rmse_matrix(np.stack(thumbs))
        for i, j in zip(*np.nonzero(np.triu(distances <= cfg.duplicate_rmse, k=1))):
            if rows[i]["split"] != rows[j]["split"]:
                near_dupes.append(f"{rows[i]['path']} <-> {rows[j]['path']} (rmse {distances[i, j]:.1f})")
    return {"shared_groups": shared_groups, "cross_split_near_duplicates": near_dupes}


def split_dataset(cfg: SplitConfig, log: Callable[[str], None] = print) -> dict:
    """Copy <input>/<class>/*.jpg into <output>/<split>/<class>/ and return a summary dict."""
    _validate_ratios(cfg.ratios)
    input_dir, output_dir, reports_dir = map(Path, (cfg.input_dir, cfg.output_dir, cfg.reports_dir))
    rng = random.Random(cfg.seed)
    plan: list[dict] = []
    for class_name in cfg.class_names:
        class_dir = input_dir / class_name
        files = sorted(p for p in class_dir.glob("*") if p.suffix.lower() in _IMAGE_SUFFIXES) if class_dir.is_dir() else []
        if not files:
            raise FileNotFoundError(f"No images for class '{class_name}' in {class_dir}")
        by_group: dict[str, list[Path]] = defaultdict(list)
        for f in files:
            by_group[group_of(f)].append(f)
        if len(by_group) < len(cfg.ratios):
            raise ValueError(
                f"Class '{class_name}' has only {len(by_group)} sample group(s); at least {len(cfg.ratios)} "
                "are needed so that every split gets one. Capture more physical samples."
            )
        assignment = assign_groups_to_splits({g: len(v) for g, v in by_group.items()}, cfg.ratios, rng)
        for group, group_files in sorted(by_group.items()):
            for f in group_files:
                plan.append({
                    "split": assignment[group],
                    "class": class_name,
                    "group": group,
                    "path": f"{assignment[group]}/{class_name}/{f.name}",
                    "source": f,
                })

    prepare_output_dir(output_dir)
    for row in plan:
        target = output_dir / row["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(row["source"], target)
    log(f"Copied {len(plan)} images into {output_dir}")

    counts = {s: Counter() for s in cfg.ratios}
    groups = {s: Counter() for s in cfg.ratios}
    for row in plan:
        counts[row["split"]][row["class"]] += 1
    for split, class_name, _ in {(r["split"], r["class"], r["group"]) for r in plan}:
        groups[split][class_name] += 1

    total = len(plan)
    summary = {
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "seed": cfg.seed,
        "requested_ratios": cfg.ratios,
        "achieved_ratios": {s: round(sum(counts[s].values()) / total, 4) for s in cfg.ratios},
        "images": {s: {c: counts[s][c] for c in cfg.class_names} for s in cfg.ratios},
        "groups": {s: {c: groups[s][c] for c in cfg.class_names} for s in cfg.ratios},
        "total_images": total,
    }
    if cfg.audit:
        log("Running leakage audit ...")
        summary["audit"] = _audit(plan, output_dir, cfg)

    reports_dir.mkdir(parents=True, exist_ok=True)
    with open(reports_dir / "split_manifest.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["split", "class", "group", "path"])
        writer.writeheader()
        writer.writerows({k: row[k] for k in ("split", "class", "group", "path")} for row in plan)
    (reports_dir / "split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def format_split_summary(summary: dict, class_names: Sequence[str] = CLASS_NAMES) -> str:
    splits = list(summary["images"])
    header = f"{'class':<16}" + "".join(f"{s:>14}" for s in splits) + f"{'total':>8}"
    lines = [header, "-" * len(header)]
    for c in class_names:
        cells = "".join(
            f"{summary['images'][s][c]:>7} ({summary['groups'][s][c]:>2} g)" for s in splits
        )
        lines.append(f"{c:<16}{cells}{sum(summary['images'][s][c] for s in splits):>8}")
    lines.append("-" * len(header))
    lines.append(
        f"{'total':<16}" + "".join(f"{sum(summary['images'][s].values()):>14}" for s in splits)
        + f"{summary['total_images']:>8}"
    )
    lines.append(
        "achieved ratios: " + ", ".join(f"{s} {summary['achieved_ratios'][s]:.1%}" for s in splits)
        + f"   (seed {summary['seed']}, g = sample groups)"
    )
    audit = summary.get("audit")
    if audit is not None:
        if not audit["shared_groups"] and not audit["cross_split_near_duplicates"]:
            lines.append("leakage audit: PASS (no shared sample groups, no near-duplicates across splits)")
        else:
            lines.append("leakage audit: CHECK")
            lines += [f"  ! shared group: {x}" for x in audit["shared_groups"]]
            lines += [f"  ! near-duplicate across splits: {x}" for x in audit["cross_split_near_duplicates"]]
    return "\n".join(lines)
