"""Dataset statistics and figures for the README and the report (pipeline step 3).

    python scripts/dataset_report.py               # reads data/splits (or data/processed) + dataset/capture_log.csv

Writes to results/dataset/:
    class_distribution.png   images per class, stacked by split
    sample_grid.png          random examples of every class (from different physical samples)
    dataset_summary.md       tables ready to paste into the report
    dataset_summary.json     the same numbers, machine-readable (used by the website)
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from fabricscan import config  # noqa: E402
from fabricscan.splitting import group_of, group_sources  # noqa: E402

# Chart tokens: validated reference palette (dataviz skill). Slots 1-3 are safe as an all-pairs set.
SURFACE, INK, INK_2, MUTED, GRID, BASELINE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
FONT = ["Segoe UI", "DejaVu Sans", "sans-serif"]


def count_images(root: Path, class_names) -> dict[str, dict[str, list[Path]]]:
    """{split: {class: [files]}}; a processed folder (no splits) is returned as the single split 'all'."""
    def files(d: Path) -> list[Path]:
        return sorted(p for p in d.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")) if d.is_dir() else []

    if all((root / s).is_dir() for s in config.SPLITS):
        return {s: {c: files(root / s / c) for c in class_names} for s in config.SPLITS}
    return {"all": {c: files(root / c) for c in class_names}}


SOURCES = group_sources(config.DATASET_DIR / "SOURCES.csv")


def source_of(group: str) -> str:
    """Where a sample came from (dataset/SOURCES.csv, written by scripts/import_sources.py)."""
    return SOURCES.get(group, "Own photos")


def md_table(headers: list[str], rows: list[list]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---:" if i else "---" for i in range(len(headers))) + "|"]
    out += ["| " + " | ".join(str(v) for v in row) + " |" for row in rows]
    return out


def plot_distribution(counts: dict[str, dict[str, int]], class_names, path: Path) -> None:
    plt.rcParams["font.family"] = FONT
    splits = list(counts)
    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(class_names) + 1.4), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    y = np.arange(len(class_names))[::-1]
    left = np.zeros(len(class_names))
    for i, split in enumerate(splits):
        values = np.array([counts[split][c] for c in class_names], dtype=float)
        ax.barh(y, values, left=left, height=0.56, color=SERIES[i], edgecolor=SURFACE, linewidth=0.75,
                label=split if len(splits) > 1 else None)
        left += values
    span = max(left.max(), 1)
    for yi, total in zip(y, left):
        ax.text(total + span * 0.012, yi, f"{int(total)}", va="center", ha="left", fontsize=8, color=INK)
    ax.set_yticks(y, [config.DISPLAY_NAMES.get(c, c) for c in class_names], fontsize=8.5, color=INK)
    ax.tick_params(axis="x", colors=MUTED, labelsize=8, length=0)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.set_xlim(0, span * 1.12)
    ax.set_xlabel("Images", color=INK_2, fontsize=8)
    title = "Images per class and split" if len(splits) > 1 else "Images per class (cleaned)"
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=22 if len(splits) > 1 else 10, fontweight="semibold")
    if len(splits) > 1:
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=len(splits), frameon=False, fontsize=8,
                  labelcolor=INK_2, handlelength=1.0, handleheight=0.8, borderaxespad=0.2, columnspacing=1.4)
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def plot_samples(files_by_class: dict[str, list[Path]], class_names, n: int, seed: int, path: Path) -> None:
    """n examples per class, each from a different physical sample where possible."""
    plt.rcParams["font.family"] = FONT
    rng = random.Random(seed)
    fig, axes = plt.subplots(len(class_names), n, figsize=(n * 1.45 + 1.7, len(class_names) * 1.45), dpi=150,
                             squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for r, c in enumerate(class_names):
        by_group: dict[str, list[Path]] = defaultdict(list)
        for f in files_by_class.get(c, []):
            by_group[group_of(f)].append(f)
        groups = sorted(by_group)
        rng.shuffle(groups)
        picks = [rng.choice(by_group[g]) for g in groups[:n]]
        for k in range(n):
            ax = axes[r, k]
            ax.axis("off")
            if k < len(picks):
                with Image.open(picks[k]) as img:
                    ax.imshow(img.convert("RGB"))
        axes[r, 0].text(-0.08, 0.5, config.DISPLAY_NAMES.get(c, c), transform=axes[r, 0].transAxes, ha="right",
                        va="center", fontsize=9, color=INK)
    fig.suptitle("Example images per class", x=0.02, ha="left", fontsize=11, color=INK, fontweight="semibold")
    fig.subplots_adjust(left=0.2, wspace=0.04, hspace=0.06, top=0.93)
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def analyse_capture_log(path: Path, class_names, groups_by_class: dict[str, set[str]]):
    """Markdown lines, JSON stats and warnings derived from dataset/capture_log.csv."""
    if not path.is_file():
        return ["_No capture log found._"], {}, []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = [r for r in csv.DictReader(fh) if any((v or "").strip() for v in r.values())]
    if not rows:
        return ["_capture_log.csv is empty. Fill it in to get fabric, lighting and size statistics._"], {}, []
    clean = lambda v: (v or "").strip()  # noqa: E731
    warnings, lines, stats = [], [], {"samples_logged": len(rows)}
    bad = [r for r in rows if clean(r.get("class")) not in class_names]
    if bad:
        warnings.append(f"{len(bad)} capture-log row(s) have an unknown class value.")
    rows = [r for r in rows if clean(r.get("class")) in class_names]

    def crosstab(column: str, skip_good: bool = False) -> list[str]:
        table = Counter((clean(r.get(column)) or "(blank)", clean(r["class"])) for r in rows
                        if not (skip_good and r["class"] == "good"))
        values = sorted({v for v, _ in table})
        cols = [c for c in class_names if not (skip_good and c == "good")]
        stats[column] = {v: {c: table[(v, c)] for c in cols} for v in values}
        return md_table([column] + cols, [[v] + [table[(v, c)] for c in cols] for v in values])

    for column, title, skip in (("fabric_type", "Fabric types (samples per class)", False),
                                ("lighting", "Lighting (samples per class)", False),
                                ("defect_size", "Defect size (samples per class)", True)):
        lines += [f"### {title}", ""] + crosstab(column, skip) + [""]

    coverage: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        if clean(r.get("fabric_id")):
            coverage[clean(r["fabric_id"])].add(r["class"])
    if coverage:
        lines += ["### Fabric coverage across classes", ""]
        lines += md_table(["fabric_id", "classes", "missing from"],
                          [[f, f"{len(cs)}/{len(class_names)}", ", ".join(c for c in class_names if c not in cs) or "-"]
                           for f, cs in sorted(coverage.items())])
        lines.append("")
        weak = [f for f, cs in coverage.items() if len(cs) <= len(class_names) // 2]
        if weak:
            warnings.append(f"Fabrics used in half the classes or fewer (shortcut-learning risk): {', '.join(sorted(weak))}")
        stats["fabric_coverage"] = {f: sorted(cs) for f, cs in coverage.items()}

    logged = defaultdict(set)
    for r in rows:
        logged[r["class"]].add(clean(r.get("sample_id")))
    missing = {c: sorted(g for g in groups_by_class.get(c, set()) if not g.startswith("auto") and g not in logged[c])
               for c in class_names}
    if any(missing.values()):
        warnings.append("Sample folders without a capture-log row: "
                        + "; ".join(f"{c}: {len(m)}" for c, m in missing.items() if m))
    return lines, stats, warnings


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data", type=Path, default=None, help="splits or processed folder (default: data/splits, "
                                                          "falling back to data/processed)")
    p.add_argument("--reports", type=Path, default=config.REPORTS_DIR, help="where clean_summary.json lives")
    p.add_argument("--capture-log", type=Path, default=config.CAPTURE_LOG)
    p.add_argument("--out", type=Path, default=config.RESULTS_DIR / "dataset")
    p.add_argument("--samples-per-class", type=int, default=5)
    p.add_argument("--seed", type=int, default=config.SEED)
    args = p.parse_args(argv)

    data = args.data or (config.SPLITS_DIR if config.SPLITS_DIR.is_dir() else config.PROCESSED_DIR)
    class_names = list(config.CLASS_NAMES)
    files = count_images(data, class_names)
    if not any(f for split in files.values() for f in split.values()):
        print(f"No images found in {data}. Run clean_dataset.py (and split_dataset.py) first.")
        return 1
    args.out.mkdir(parents=True, exist_ok=True)
    counts = {s: {c: len(v) for c, v in per.items()} for s, per in files.items()}
    groups = {c: {group_of(f) for per in files.values() for f in per[c]} for c in class_names}
    totals = {c: sum(counts[s][c] for s in counts) for c in class_names}
    grand = sum(totals.values())

    plot_distribution(counts, class_names, args.out / "class_distribution.png")
    sample_source = files["train"] if "train" in files else files["all"]
    plot_samples(sample_source, class_names, args.samples_per_class, args.seed, args.out / "sample_grid.png")

    splits = list(counts)
    sources = Counter(source_of(group_of(f)) for per in files.values() for fl in per.values() for f in fl)
    sources_per_class = {c: dict(Counter(source_of(group_of(f)) for per in files.values() for f in per[c]))
                         for c in class_names}
    # Paths relative to the project, so no local folder names end up in the published files.
    shown = data.resolve().relative_to(config.PROJECT_ROOT).as_posix() if data.resolve().is_relative_to(config.PROJECT_ROOT) else data.name
    lines = ["# Dataset summary", "", f"Folder: `{shown}`. Origin and licence of every image: `dataset/SOURCES.csv`.", "", "## Images per class", ""]
    rows = [[config.DISPLAY_NAMES[c]] + [counts[s][c] for s in splits] + [totals[c], len(groups[c])] for c in class_names]
    rows.append(["**Total**"] + [f"**{sum(counts[s].values())}**" for s in splits] + [f"**{grand}**",
                                                                                     f"**{sum(len(g) for g in groups.values())}**"])
    lines += md_table(["Class"] + [s.capitalize() for s in splits] + ["Total", "Physical samples"], rows) + [""]
    if len(splits) > 1:
        lines.append("Split share: " + ", ".join(f"{s} {sum(counts[s].values()) / grand:.1%}" for s in splits)
                     + f" (group-aware, seed {args.seed}).")
        lines.append("")
    imbalance = max(totals.values()) / max(min(totals.values()), 1)
    lines.append(f"Class balance (largest / smallest): **{imbalance:.2f}**.")
    lines.append("")
    source_names = sorted(sources)
    lines += ["## Image sources", ""]
    lines += md_table(["Class"] + source_names,
                      [[config.DISPLAY_NAMES[c]] + [sources_per_class[c].get(s, 0) for s in source_names] for c in class_names]
                      + [["**Total**"] + [f"**{sources[s]}**" for s in source_names]])
    lines.append("")

    clean_summary_path = args.reports / "clean_summary.json"
    clean_stats = {}
    if clean_summary_path.is_file():
        clean_stats = json.loads(clean_summary_path.read_text(encoding="utf-8"))
        removed = Counter()
        for per in clean_stats["per_class"].values():
            removed.update(per["removed"])
        t = clean_stats["totals"]
        lines += ["## Cleaning", "", f"{t['files_found']} files found, {t['kept']} kept, {t['removed']} removed, "
                  f"{t['flagged_for_review']} flagged for manual review.", ""]
        if removed:
            lines += md_table(["Removal reason", "Files"], [[k, v] for k, v in sorted(removed.items())]) + [""]

    log_lines, log_stats, warnings = analyse_capture_log(args.capture_log, class_names, groups)
    lines += ["## Capture conditions (from capture_log.csv)", ""] + log_lines
    if warnings:
        lines += ["## Warnings", ""] + [f"- {w}" for w in warnings] + [""]
    (args.out / "dataset_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary = {
        "source": shown,
        "class_names": class_names,
        "display_names": config.DISPLAY_NAMES,
        "images": counts,
        "totals": totals,
        "total_images": grand,
        "physical_samples": {c: len(g) for c, g in groups.items()},
        "imbalance_ratio": round(imbalance, 3),
        "sources": dict(sources),
        "sources_per_class": sources_per_class,
        "capture_log": log_stats,
        "warnings": warnings,
    }
    (args.out / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n".join(lines[6:6 + len(rows) + 2]))
    for w in warnings:
        print(f"  ! {w}")
    print(f"\nFigures and tables written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
