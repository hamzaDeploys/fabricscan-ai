"""Build dataset/ from the collected image sources and write dataset/SOURCES.csv.

    python scripts/import_sources.py                      # rebuilds dataset/<class>/<group>/*.jpg
    python scripts/import_sources.py --collect "C:/Users/<you>/Downloads/data collect"

Sources (all downloads live in data/external/kaggle/<owner>__<dataset>/archive.zip):

* **Hand-picked folder** ("data collect"): images already sorted into class folders. They come from the Kaggle
  dataset nexuswho/fabric-defects-dataset (CC BY 4.0), a few web images and a few AI-generated images; the two
  SOURCES_*.csv files in that folder record the origin of each one. `_review/` is skipped.
* **Kaggle pulakdroy/fabric-defect-without-augmentation** (CC BY 4.0, YOLO boxes): Hole -> hole,
  Thick-Missing yarn and Knot-slub -> broken_thread, Spot -> stain.
* **Kaggle anmol055/clothes-staindefect-detection** (CC BY 4.0, YOLO boxes): stains on worn clothes -> stain. Its
  images without boxes are whole clean garments; square crops from their centre become `good` close-ups, so `good`
  and `stain` share the same garments and photo style.
* **Kaggle hyungjung/stitchingnet-dataset** (CC BY-NC 4.0, 224 px seam close-ups, upscaled to 256 px):
  "2. Broken stitch" -> broken_thread, "0. Normal" -> good.
* **Kaggle rajkumarl/people-clothing-segmentation** (CC0): photos of people with a clothing mask per pixel. A square
  crop inside the largest upper-body garment becomes `good`: clean shirts, tops and jackets as they are worn, which
  is what a phone camera sees in the live demo. Without them, every photo of a worn garment in the data was a stain.

Detection boxes become square crops around the defect (2-3x the box, at least 256 px where the image allows), so
the defect is clearly visible at the model's 224 px input. A crop that would also contain a box of another class is
skipped. Every view of one original image shares a group folder, so the split keeps it in one subset; Roboflow
exports name augmented copies "<stem>_jpg.rf.<hash>", and those share the group of their original as well.
Images with more than MAX_BOXES boxes of one class (dense pattern labels) are skipped. Per-source caps keep the
classes balanced. Web images (unknown licence) and AI-generated images are left out, so the whole dataset can be
published as a zip with the licence of every file.
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import shutil
import sys
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fabricscan import config  # noqa: E402
from fabricscan.fsutils import safe_name  # noqa: E402

KAGGLE_DIR = config.DATA_DIR / "external" / "kaggle"
MIN_CROP = 256
MAX_BOXES = 6


@dataclass
class YoloSource:
    ref: str  # Kaggle owner/dataset
    licence: str
    mapping: dict[str, str]  # source class name -> FabricScan class
    caps: dict[str, int]  # max crops per FabricScan class
    names: list[str] | None = None  # class names when the export has no data.yaml
    crops_per_box: int = 1
    clean_class: str | None = None  # images without boxes are clean: centre crops become this class
    clean_cap: int = 0


@dataclass
class MaskSource:
    ref: str
    licence: str
    images: str  # folder with the photos
    masks: str  # folder with one label-id mask per photo (same sort order)
    garment_ids: tuple[int, ...]  # mask ids that count as the garment to crop
    target: str  # FabricScan class
    cap: int


@dataclass
class FolderSource:
    ref: str
    licence: str
    mapping: dict[str, str]  # folder name (last path part) -> FabricScan class
    caps: dict[str, int] = field(default_factory=dict)


YOLO_SOURCES = [
    YoloSource("pulakdroy/fabric-defect-without-augmentation", "CC BY 4.0",
               {"Hole": "hole", "Thick-Missing yarn": "broken_thread", "Knot-slub": "broken_thread", "Spot": "stain"},
               {"hole": 220, "broken_thread": 140, "stain": 50}),
    YoloSource("anmol055/clothes-staindefect-detection", "CC BY 4.0", {"stain": "stain"}, {"stain": 140},
               names=["stain"], clean_class="good", clean_cap=100),
]
# blazer, blouse, cardigan, coat, dress, hoodie, jacket, jumper, shirt, suit, sweater, sweatshirt, t-shirt, top, vest
UPPER_BODY = (4, 5, 11, 13, 14, 22, 24, 26, 38, 46, 48, 49, 51, 54, 55)
MASK_SOURCES = [
    MaskSource("rajkumarl/people-clothing-segmentation", "CC0", "png_images/IMAGES", "png_masks/MASKS",
               UPPER_BODY, "good", 110),
]
COLLECT_CAPS = {"good": 100, "stain": 100}
# Plain grey cotton without defects, shot with the same camera as the grey stain photos: always kept, because
# without them the model learns "plain light cotton = stain".
COLLECT_ALWAYS = re.compile(r"^fd_defect_free_\d+\.\w+$")
# Held out of the dataset on purpose: shown on the website as sample images (assets/held_out_samples/).
HELD_OUT = {"fd_defect_free_516fdcea757340d70914292471.jpg", "fd_defect_free_20.jpg"}
# Near-black industrial shots from nexuswho/fabric-defects-dataset: the defect is barely visible at 224 px.
COLLECT_SKIP = re.compile(r"^(fd_(hole_\d+\.\w+$|horizontal_|vertical_)|web_|ai_)")
FOLDER_SOURCES = [
    FolderSource("hyungjung/stitchingnet-dataset", "CC BY-NC 4.0",
                 {"2. Broken stitch": "broken_thread", "0. Normal": "good"}, {"broken_thread": 100, "good": 40}),
]


def archive_dir(ref: str) -> Path:
    return KAGGLE_DIR / ref.replace("/", "__")


def extract(ref: str) -> Path:
    """Unpack archive.zip next to itself once; refuses members that would escape the folder."""
    folder = archive_dir(ref)
    out = folder / "extracted"
    if out.is_dir():
        return out
    tmp = folder / "_extracting"
    shutil.rmtree(tmp, ignore_errors=True)
    with zipfile.ZipFile(folder / "archive.zip") as zf:
        for member in zf.infolist():
            target = (tmp / member.filename).resolve()
            if not target.is_relative_to(tmp.resolve()):
                raise ValueError(f"unsafe path in {ref}: {member.filename}")
        zf.extractall(tmp)
    tmp.rename(out)
    return out


def original_stem(name: str) -> str:
    """Roboflow names augmented copies '<stem>_jpg.rf.<hash>.jpg'; all of them belong to one original."""
    return re.sub(r"\.rf\.[0-9a-f]+$", "", Path(name).stem)


def read_yolo_names(root: Path, source: YoloSource) -> list[str]:
    if source.names:
        return source.names
    yaml = next(root.rglob("data.yaml"), None)
    if yaml is None:
        raise FileNotFoundError(f"{source.ref}: no data.yaml; set names= in YOLO_SOURCES")
    match = re.search(r"names:\s*\[(.*?)\]", yaml.read_text(encoding="utf-8"), re.S)
    if match:
        return [n.strip().strip("'\"") for n in match.group(1).split(",")]
    return re.findall(r"^\s*-\s*['\"]?(.+?)['\"]?\s*$", yaml.read_text(encoding="utf-8").split("names:")[1], re.M)


def crop_around(rng: random.Random, size: tuple[int, int], box: tuple[float, float, float, float],
                others: list[tuple[float, float, float, float]]) -> tuple[int, int, int] | None:
    """A square (x, y, side) that holds the box with context and touches none of the other-class boxes."""
    w, h = size
    x0, y0, x1, y1 = box
    limit = min(w, h)
    for _ in range(20):
        side = max(x1 - x0, y1 - y0) * rng.uniform(2.0, 3.0)
        side = int(min(limit, max(side, MIN_CROP, 0.3 * limit)))
        lo_x, hi_x = max(0, int(x1) - side), min(int(x0), w - side)
        lo_y, hi_y = max(0, int(y1) - side), min(int(y0), h - side)
        x = rng.randint(lo_x, hi_x) if lo_x <= hi_x else max(0, min(w - side, int((x0 + x1 - side) / 2)))
        y = rng.randint(lo_y, hi_y) if lo_y <= hi_y else max(0, min(h - side, int((y0 + y1 - side) / 2)))
        if not any(ox0 < x + side and ox1 > x and oy0 < y + side and oy1 > y for ox0, oy0, ox1, oy1 in others):
            return x, y, side
    return None


def save(img: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(path, "JPEG", quality=95)


def import_yolo(source: YoloSource, dataset: Path, rng: random.Random) -> list[dict]:
    root = extract(source.ref)
    names = read_yolo_names(root, source)
    short = safe_name(source.ref.split("/")[1], "kaggle")[:12]
    candidates: dict[str, list[tuple]] = defaultdict(list)  # class -> (image, box, other-class boxes)
    for label_file in sorted(p for p in root.rglob("*.txt") if "labels" in p.parts):
        parts = list(label_file.parent.parts)
        parts[len(parts) - 1 - parts[::-1].index("labels")] = "images"  # .../labels/<split>/x.txt -> .../images/<split>/
        images = sorted(Path(*parts).glob(f"{label_file.stem}.*"))
        if not images:
            continue
        text = label_file.read_text(encoding="utf-8")
        if source.clean_class and not text.strip():
            candidates["_clean"].append((images[0], None, []))
            continue
        boxes = []
        for line in text.split("\n"):
            values = line.split()
            if len(values) < 5 or names[int(values[0])] not in source.mapping:
                continue
            coords = [float(v) for v in values[1:]]
            if len(coords) > 4:  # a polygon x1 y1 x2 y2 ... -> its bounding box
                xs, ys = coords[0::2], coords[1::2]
                coords = [(min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, max(xs) - min(xs), max(ys) - min(ys)]
            boxes.append((source.mapping[names[int(values[0])]], coords))
        per_class = Counter(c for c, _ in boxes)
        for cls, b in boxes:
            if per_class[cls] <= MAX_BOXES:
                candidates[cls].append((images[0], b, [o for c, o in boxes if c != cls]))
    rows = []
    clean = candidates.pop("_clean", [])
    for image_path, _, _ in rng.sample(clean, min(source.clean_cap, len(clean))):
        with Image.open(image_path) as im:
            im = ImageOps.exif_transpose(im)
            w, h = im.size
            side = min(min(w, h), max(MIN_CROP, int(min(w, h) * rng.uniform(0.3, 0.6))))
            x = max(0, min(w - side, int(w / 2 - side / 2 + rng.uniform(-0.1, 0.1) * w)))
            y = max(0, min(h - side, int(h / 2 - side / 2 + rng.uniform(-0.1, 0.1) * h)))
            group = f"kg_{short}_{safe_name(original_stem(image_path.name), 'img')}"[:60]
            out = dataset / source.clean_class / group / "centre.jpg"
            if out.exists():  # the export repeats some file names across its train/val/test folders
                continue
            save(im.crop((x, y, x + side, y + side)), out)
        rows.append({"file": out.relative_to(dataset).as_posix(), "class": source.clean_class,
                     "source": f"Kaggle: {source.ref}", "licence": source.licence,
                     "original": image_path.relative_to(root).as_posix(),
                     "note": f"centre crop {x},{y},{side} of an image without defect boxes"})
    for cls, items in candidates.items():
        rng.shuffle(items)
        used_originals: Counter = Counter()
        for image_path, b, others in items:
            if sum(used_originals.values()) >= source.caps.get(cls, 0):
                break
            stem = original_stem(image_path.name)
            if used_originals[stem] >= source.crops_per_box:  # augmented copies would only add near-duplicates
                continue
            with Image.open(image_path) as im:
                im = ImageOps.exif_transpose(im)
                w, h = im.size
                to_px = lambda c: (  # noqa: E731
                    (c[0] - c[2] / 2) * w, (c[1] - c[3] / 2) * h, (c[0] + c[2] / 2) * w, (c[1] + c[3] / 2) * h)
                crop = crop_around(rng, (w, h), to_px(b), [to_px(o) for o in others])
                if crop is None:
                    continue
                x, y, side = crop
                group = f"kg_{short}_{safe_name(stem, 'img')}"[:60]
                out = dataset / cls / group / f"crop{used_originals[stem] + 1}.jpg"
                save(im.crop((x, y, x + side, y + side)), out)
            used_originals[stem] += 1
            rows.append({"file": out.relative_to(dataset).as_posix(), "class": cls, "source": f"Kaggle: {source.ref}",
                         "licence": source.licence, "original": image_path.relative_to(root).as_posix(),
                         "note": f"crop {x},{y},{side} around a labelled box"})
    return rows


def import_masked(source: MaskSource, dataset: Path, rng: random.Random) -> list[dict]:
    """One square close-up inside the largest garment region of each photo (skips photos where it is small)."""
    root = extract(source.ref)
    short = safe_name(source.ref.split("/")[1], "kaggle")[:12]
    pairs = list(zip(sorted((root / source.images).glob("*.png")), sorted((root / source.masks).glob("*.png"))))
    rng.shuffle(pairs)
    rows = []
    for image_path, mask_path in pairs:
        if len(rows) >= source.cap:
            break
        with Image.open(mask_path) as m:
            garment = np.isin(np.asarray(m), source.garment_ids)
        ys, xs = np.nonzero(garment)
        if len(xs) < 0.06 * garment.size:
            continue
        x0, x1, y0, y1 = np.percentile(xs, 10), np.percentile(xs, 90), np.percentile(ys, 10), np.percentile(ys, 90)
        with Image.open(image_path) as im:
            im = ImageOps.exif_transpose(im)
            w, h = im.size
            side = int(min(w, h, max(160, min(x1 - x0, y1 - y0) * rng.uniform(0.7, 1.0))))
            x = int(max(0, min(w - side, (x0 + x1) / 2 - side / 2)))
            y = int(max(0, min(h - side, (y0 + y1) / 2 - side / 2)))
            group = f"kg_{short}_{safe_name(image_path.stem, 'img')}"[:60]
            out = dataset / source.target / group / "garment.jpg"
            save(im.crop((x, y, x + side, y + side)), out)
        rows.append({"file": out.relative_to(dataset).as_posix(), "class": source.target, "source": f"Kaggle: {source.ref}",
                     "licence": source.licence, "original": image_path.relative_to(root).as_posix(),
                     "note": f"crop {x},{y},{side} inside the worn garment (mask)"})
    return rows


def import_folders(source: FolderSource, dataset: Path, rng: random.Random) -> list[dict]:
    root = extract(source.ref)
    short = safe_name(source.ref.split("/")[1], "kaggle")[:12]
    by_class: dict[str, list[Path]] = defaultdict(list)
    for folder in sorted(p for p in root.rglob("*") if p.is_dir() and p.name in source.mapping):
        by_class[source.mapping[folder.name]] += sorted(f for f in folder.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png"))
    rows = []
    for cls, files in by_class.items():
        for f in rng.sample(files, min(source.caps.get(cls, len(files)), len(files))):
            group = f"kg_{short}_{safe_name(f.parent.parent.name[:12] + '_' + f.stem, 'img')}"[:60]
            out = dataset / cls / group / f"{safe_name(f.stem, 'img')}.jpg"
            with Image.open(f) as im:
                im, note = ImageOps.exif_transpose(im), ""
                if min(im.size) < MIN_CROP:  # the cleaning step rejects images below MIN_CROP pixels
                    note = f"upscaled from {im.width}x{im.height}"
                    scale = MIN_CROP / min(im.size)
                    im = im.resize((round(im.width * scale), round(im.height * scale)), Image.Resampling.LANCZOS)
                save(im, out)
            rows.append({"file": out.relative_to(dataset).as_posix(), "class": cls, "source": f"Kaggle: {source.ref}",
                         "licence": source.licence, "original": f.relative_to(root).as_posix(), "note": note})
    return rows


def import_collect(collect: Path, dataset: Path, rng: random.Random) -> list[dict]:
    """Copy the hand-picked class folders; origin and licence come from the SOURCES_*.csv files in that folder."""
    origin = {}
    for log in collect.glob("SOURCES_*.csv"):
        for r in csv.DictReader(open(log, encoding="utf-8")):
            origin[r["file"]] = r
    rows = []
    for cls in config.CLASS_NAMES:
        folder = collect / cls
        files = sorted(f for f in folder.glob("*") if f.is_file() and not COLLECT_SKIP.match(f.name)) if folder.is_dir() else []
        files = [f for f in files if f.name not in HELD_OUT]
        always = [f for f in files if COLLECT_ALWAYS.match(f.name)]
        rest = [f for f in files if f not in always]
        cap = COLLECT_CAPS.get(cls, len(files))
        picked = always + rng.sample(rest, max(0, min(len(rest), cap - len(always))))
        for f in sorted(picked):
            info = origin.get(f"{cls}/{f.name}", {})
            group = f"dc_{safe_name(f.stem, 'img')}"[:60]
            out = dataset / cls / group / f"{safe_name(f.stem, 'img')}{f.suffix.lower()}"
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, out)
            rows.append({"file": out.relative_to(dataset).as_posix(), "class": cls,
                         "source": info.get("source", "hand-picked"), "licence": info.get("licence", ""),
                         "original": info.get("original", f.name), "note": info.get("note", "")})
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--collect", type=Path, default=Path.home() / "Downloads" / "data collect")
    p.add_argument("--dataset", type=Path, default=config.DATASET_DIR)
    p.add_argument("--seed", type=int, default=config.SEED)
    args = p.parse_args(argv)
    rng = random.Random(args.seed)

    # Folders whose name starts with kg_ or dc_ are created only by this script: rebuild them from scratch.
    for cls in config.CLASS_NAMES:
        for old in list((args.dataset / cls).glob("kg_*")) + list((args.dataset / cls).glob("dc_*")):
            shutil.rmtree(old)

    rows = import_collect(args.collect, args.dataset, rng)
    for source in YOLO_SOURCES:
        rows += import_yolo(source, args.dataset, rng)
    for source in FOLDER_SOURCES:
        rows += import_folders(source, args.dataset, rng)
    for source in MASK_SOURCES:
        rows += import_masked(source, args.dataset, rng)

    with open(args.dataset / "SOURCES.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["file", "class", "source", "licence", "original", "note"])
        writer.writeheader()
        writer.writerows(rows)
    per_class = Counter(r["class"] for r in rows)
    per_source = Counter((r["class"], r["source"]) for r in rows)
    print(f"Wrote {len(rows)} images to {args.dataset}")
    for cls in config.CLASS_NAMES:
        print(f"  {cls:<14} {per_class[cls]:5d}   " +
              ", ".join(f"{s.replace('Kaggle: ', '')}: {n}" for (c, s), n in sorted(per_source.items()) if c == cls))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
