"""Pack the cleaned, split dataset into one zip for download (pipeline step after split_dataset.py).

    python scripts/package_dataset.py          # writes web/public/downloads/fabricscan-dataset.zip (+ .json manifest)

The zip holds train/ val/ test/ with one folder per class (256 x 256 JPEGs, exactly what the models were trained and
tested on), SOURCES.csv with the origin and licence of every image, and a README.txt with the licences to respect.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fabricscan import config  # noqa: E402
from fabricscan.fsutils import safe_name  # noqa: E402
from fabricscan.splitting import group_of, group_sources  # noqa: E402

KAGGLE_TITLES = {
    "nexuswho/fabric-defects-dataset": "Fabric Defects Dataset",
    "pulakdroy/fabric-defect-without-augmentation": "Fabric Defect (without augmentation)",
    "anmol055/clothes-staindefect-detection": "Clothes Stain/Defect Detection",
    "hyungjung/stitchingnet-dataset": "StitchingNet",
    "rajkumarl/people-clothing-segmentation": "People Clothing Segmentation",
}

README = """FabricScan AI dataset
=====================

{total} images, 256 x 256 px, {n} classes: {classes}.
Split by source image (group-aware, seed {seed}) into train / val / test; no source image appears in two splits.

{table}

File names: <group>__<original name>.jpg. The group is the source image; all views of it share one split.

Sources and licences (SOURCES.csv lists every original file):
{sources}

The images keep the licences of their sources. CC BY 4.0: give credit to the dataset authors.
CC BY-NC 4.0 (StitchingNet): credit the authors and do not use commercially.
CC0 (People Clothing Segmentation): no conditions.
Built for the FabricScan AI university project with scripts/import_sources.py, clean_dataset.py and split_dataset.py.
"""


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--splits", type=Path, default=config.SPLITS_DIR)
    p.add_argument("--out", type=Path, default=config.PROJECT_ROOT / "web" / "public" / "downloads" / "fabricscan-dataset.zip")
    args = p.parse_args(argv)
    if not (args.splits / "train").is_dir():
        print(f"No split found in {args.splits}. Run clean_dataset.py and split_dataset.py first.")
        return 1

    sources_csv = config.DATASET_DIR / "SOURCES.csv"
    sources = group_sources(sources_csv)
    files = sorted(f for s in config.SPLITS for c in config.CLASS_NAMES for f in (args.splits / s / c).glob("*.jpg"))
    counts = Counter((f.parent.parent.name, f.parent.name) for f in files)
    table = ["class            " + "".join(f"{s:>7}" for s in config.SPLITS)]
    table += [f"{c:<17}" + "".join(f"{counts[(s, c)]:>7}" for s in config.SPLITS) for c in config.CLASS_NAMES]
    used = Counter(sources.get(group_of(f), "Own photos") for f in files)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.out, "w", compression=zipfile.ZIP_STORED) as zf:  # JPEGs do not compress further
        for f in files:
            zf.write(f, f"fabricscan-dataset/{f.relative_to(args.splits).as_posix()}")
        if sources_csv.is_file():
            keep = {g for g in (group_of(f) for f in files)}
            with open(sources_csv, encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                rows = [r for r in reader if safe_name(r["file"].split("/")[1], "sample") in keep]
            buffer = io.StringIO()
            writer = csv.DictWriter(buffer, fieldnames=reader.fieldnames or [])
            writer.writeheader()
            writer.writerows(rows)
            zf.writestr("fabricscan-dataset/SOURCES.csv", buffer.getvalue())
        zf.writestr("fabricscan-dataset/README.txt", README.format(
            total=len(files), n=len(config.CLASS_NAMES), classes=", ".join(config.CLASS_NAMES), seed=config.SEED,
            table="\n".join(table), sources="\n".join(f"  {n:4d}  {s}" for s, n in used.most_common())))

    # A small manifest next to the zip, read by the website's dataset section.
    per_source: dict[str, Counter] = defaultdict(Counter)
    for f in files:
        per_source[sources.get(group_of(f), "Own photos")][f.parent.name] += 1
    manifest = {
        "file": args.out.name,
        "size_bytes": args.out.stat().st_size,
        "images": len(files),
        "image_size": config.PROCESSED_SIZE,
        "classes": list(config.CLASS_NAMES),
        "splits": {s: {c: counts[(s, c)] for c in config.CLASS_NAMES} for s in config.SPLITS},
        "sources": [_source_entry(label, per_class) for label, per_class in
                    sorted(per_source.items(), key=lambda kv: -sum(kv[1].values()))],
    }
    args.out.with_suffix(".json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"{len(files)} images -> {args.out} ({args.out.stat().st_size / 1e6:.1f} MB) + {args.out.with_suffix('.json').name}")
    return 0


def _source_entry(label: str, per_class: Counter) -> dict:
    """'Kaggle: owner/name (licence)' -> title, Kaggle link, licence and images per class."""
    match = re.match(r"Kaggle: (\S+) \((.+)\)$", label)
    ref, licence = (match.group(1), match.group(2)) if match else ("", "")
    return {"title": KAGGLE_TITLES.get(ref, label), "ref": ref, "licence": licence,
            "url": f"https://www.kaggle.com/datasets/{ref}" if ref else "",
            "images": sum(per_class.values()), "classes": {c: per_class[c] for c in config.CLASS_NAMES if per_class[c]}}


if __name__ == "__main__":
    raise SystemExit(main())
