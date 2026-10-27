"""Step 1 of the data pipeline: validate, clean and standardise the raw phone photos.

For every file under <input>/<class>/[<sample>/]* this module:

1. rejects unsupported file types and unreadable, corrupt or truncated images;
2. applies the EXIF orientation, then drops all metadata (including GPS) when re-encoding;
3. converts to RGB and rejects images whose short side is below ``min_side``;
4. centre-crops to a square and resizes to ``out_size`` x ``out_size`` (Lanczos);
5. measures sharpness and exposure and flags outliers for manual review;
6. removes exact duplicates (SHA-256) and near duplicates (thumbnail RMSE) within a class. The same file in
   two classes is a label conflict and both copies are removed; a near-identical image in two classes is
   only flagged ``check_label``, because a before/after pair shot from a tripod is legitimate;
7. assigns every image to a physical-sample *group* (its sub-folder, or capture-time clusters for flat
   folders), which the split step uses to keep all shots of one sample in the same split.

Every decision is written to ``<reports>/clean_manifest.csv`` and summarised in ``clean_summary.json``.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import io
import json
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from PIL import Image, ImageOps

from .config import (
    CLASS_NAMES,
    GROUP_SEPARATOR,
    IMAGE_EXTENSIONS,
    MAX_IMBALANCE_RATIO,
    MIN_IMAGES_PER_CLASS,
    PROCESSED_SIZE,
)
from .fsutils import prepare_output_dir, safe_name
from .imaging import (
    center_crop_square,
    encode_jpeg,
    exposure_stats,
    gray_array,
    laplacian_variance,
    read_capture_time,
    read_orientation,
    register_heif,
    rmse_matrix,
    thumbnail,
    to_rgb,
)

QUALITY_FLAGS = ("blurry", "too_dark", "overexposed")
REVIEW_FLAGS = QUALITY_FLAGS + ("check_label",)  # copied to <reports>/flagged/<flag>/ for a manual look
MAX_SHOTS_PER_GROUP = 8


@dataclass
class CleanConfig:
    input_dir: Path
    output_dir: Path
    reports_dir: Path
    class_names: Sequence[str] = CLASS_NAMES
    out_size: int = PROCESSED_SIZE
    min_side: int = 128  # reject thumbnails / screenshots smaller than this (original pixels)
    metric_side: int = 512  # sharpness and exposure are measured at this resolution
    blur_threshold: float = 15.0  # Laplacian variance below this -> "blurry"
    dark_threshold: float = 35.0  # mean brightness below this -> "too_dark"
    overexposed_fraction: float = 0.25  # share of clipped pixels above this -> "overexposed"
    duplicate_rmse: float = 3.0  # 32x32 thumbnail RMSE (0-255): near duplicates across samples (leakage risk)
    burst_rmse: float = 1.0  # within one sample only re-saved/burst copies go; rotated/re-lit shots stay
    time_gap_s: float = 30.0  # flat folders: shots closer in time than this belong to the same sample
    drop_flagged: bool = False  # remove quality-flagged images instead of only reporting them
    workers: int = 0  # 0 = one per CPU core (max 8)
    jpeg_quality: int = 95


@dataclass
class ImageRecord:
    source: Path
    class_name: str
    folder_group: str | None  # first-level sub-folder under the class folder; None for flat files
    group: str = ""
    status: str = "kept"
    reason: str = ""
    flags: list[str] = field(default_factory=list)
    width: int = 0
    height: int = 0
    capture_time: float | None = None
    blur: float = float("nan")
    brightness: float = float("nan")
    overexposed: float = float("nan")
    sha256: str = ""
    thumb: np.ndarray | None = None
    jpeg: bytes | None = None
    output: Path | None = None

    def remove(self, reason: str) -> None:
        self.status, self.reason = "removed", reason
        self.jpeg = None


def discover(input_dir: Path, class_names: Sequence[str]) -> tuple[list[ImageRecord], list[str]]:
    """List every file under the class folders. Folders and files starting with '_' or '.' are ignored."""
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")
    warnings: list[str] = []
    known = set(class_names)
    for entry in sorted(input_dir.iterdir()):
        if entry.is_dir() and not entry.name.startswith(("_", ".")) and entry.name not in known:
            warnings.append(
                f"Unknown folder '{entry.name}' skipped (class folders are: {', '.join(class_names)})."
            )
    records: list[ImageRecord] = []
    for class_name in class_names:
        class_dir = input_dir / class_name
        if not class_dir.is_dir():
            warnings.append(f"Class folder missing: {class_dir}")
            continue
        for path in sorted(class_dir.rglob("*")):
            rel = path.relative_to(class_dir)
            if any(part.startswith(("_", ".")) for part in rel.parts) or not path.is_file():
                continue
            folder_group = rel.parts[0] if len(rel.parts) > 1 else None
            record = ImageRecord(source=path, class_name=class_name, folder_group=folder_group)
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                record.remove("unsupported_file_type")
            records.append(record)
    return records, warnings


def _analyse(task: tuple[str, int, int, int, int]) -> dict:
    """Decode, standardise and measure one image. Runs in a worker process, so it only returns plain data."""
    path_str, out_size, min_side, metric_side, jpeg_quality = task
    register_heif()
    try:
        data = Path(path_str).read_bytes()
    except OSError as exc:
        return {"status": "removed", "reason": f"unreadable ({exc.strerror or exc})"}
    result: dict = {"sha256": hashlib.sha256(data).hexdigest()}
    try:
        with Image.open(io.BytesIO(data)) as probe:
            probe.verify()  # structural check; the full decode below catches truncation
        img = Image.open(io.BytesIO(data))
        width, height = img.size
        if read_orientation(img) in (5, 6, 7, 8):
            width, height = height, width
        result.update(width=width, height=height, capture_time=read_capture_time(img))
        img.draft("RGB", (metric_side, metric_side))  # fast JPEG decode at reduced scale (>= metric_side)
        img.load()
        img = ImageOps.exif_transpose(img)
    except Exception:  # Pillow raises OSError, SyntaxError, ValueError, ... for damaged files
        heic = Path(path_str).suffix.lower() in (".heic", ".heif")
        reason = "heic_needs_pillow_heif" if heic and not register_heif() else "corrupt_or_unreadable"
        return {**result, "status": "removed", "reason": reason}

    if min(width, height) < min_side:
        return {**result, "status": "removed", "reason": f"too_small ({width}x{height})"}

    flags = []
    if max(width, height) / min(width, height) > 1.02:
        flags.append("center_cropped")
    square = center_crop_square(to_rgb(img))
    metric = square if square.width <= metric_side else square.resize(
        (metric_side, metric_side), Image.Resampling.LANCZOS
    )
    gray = gray_array(metric)
    brightness, overexposed = exposure_stats(gray)
    processed = square.resize((out_size, out_size), Image.Resampling.LANCZOS)
    return {
        **result,
        "status": "kept",
        "flags": flags,
        "blur": laplacian_variance(gray),
        "brightness": brightness,
        "overexposed": overexposed,
        "thumb": thumbnail(processed),
        "jpeg": encode_jpeg(processed, jpeg_quality),
    }


def _run_parallel(tasks: list, workers: int, log: Callable[[str], None]) -> list[dict]:
    workers = workers or min(os.cpu_count() or 1, 8)
    results: list[dict] = []
    pool = ProcessPoolExecutor(max_workers=workers) if workers > 1 and len(tasks) >= 16 else None
    try:
        iterator = pool.map(_analyse, tasks, chunksize=4) if pool else map(_analyse, tasks)
        for i, res in enumerate(iterator, 1):
            results.append(res)
            if i % 100 == 0 or i == len(tasks):
                log(f"  analysed {i}/{len(tasks)}")
    finally:
        if pool:
            pool.shutdown()
    return results


def _flag_quality(records: list[ImageRecord], cfg: CleanConfig) -> None:
    for r in records:
        if r.blur < cfg.blur_threshold:
            r.flags.append("blurry")
        if r.brightness < cfg.dark_threshold:
            r.flags.append("too_dark")
        if r.overexposed > cfg.overexposed_fraction:
            r.flags.append("overexposed")
        bad = [f for f in r.flags if f in QUALITY_FLAGS]
        if cfg.drop_flagged and bad:
            r.remove("quality:" + "+".join(bad))


def _rel(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _remove_duplicates(records: list[ImageRecord], cfg: CleanConfig) -> None:
    """Exact duplicates by SHA-256, then near duplicates by thumbnail RMSE.

    Same class, different samples (RMSE <= duplicate_rmse): the later copy is removed, otherwise it could
    leak between splits. Same sample: only near-identical copies (RMSE <= burst_rmse) are removed, because a
    rotated or re-lit shot of a plain fabric can look alike and is still useful. Different classes: both are
    flagged ``check_label`` and kept for review. Requires groups to be assigned first.
    """
    first_by_hash: dict[str, ImageRecord] = {}
    for r in records:
        if r.status != "kept":
            continue
        original = first_by_hash.get(r.sha256)
        if original is None:
            first_by_hash[r.sha256] = r
        elif original.class_name != r.class_name:
            r.remove(f"label_conflict_with:{_rel(original.source, cfg.input_dir)}")
            original.remove(f"label_conflict_with:{_rel(r.source, cfg.input_dir)}")
        else:
            r.remove(f"exact_duplicate_of:{_rel(original.source, cfg.input_dir)}")

    kept = [r for r in records if r.status == "kept"]
    if len(kept) < 2:
        return
    distances = rmse_matrix(np.stack([r.thumb for r in kept]))
    rows, cols = np.nonzero(np.triu(distances <= cfg.duplicate_rmse, k=1))
    for i, j in zip(rows.tolist(), cols.tolist()):
        a, b = kept[i], kept[j]
        if a.status != "kept" or b.status != "kept":
            continue
        if a.class_name != b.class_name:
            for r in (a, b):
                if "check_label" not in r.flags:
                    r.flags.append("check_label")
        elif a.group != b.group or distances[i, j] <= cfg.burst_rmse:
            b.remove(f"near_duplicate_of:{_rel(a.source, cfg.input_dir)} (rmse {distances[i, j]:.1f})")


def _assign_groups(records: list[ImageRecord], cfg: CleanConfig) -> None:
    """Sub-folder name = group. Flat files are clustered by EXIF capture time (gap <= time_gap_s)."""
    for class_name in cfg.class_names:
        recs = [r for r in records if r.class_name == class_name and r.status == "kept"]
        for r in recs:
            if r.folder_group is not None:
                r.group = safe_name(r.folder_group, "sample")
        flat = [r for r in recs if r.folder_group is None]
        timed = sorted((r for r in flat if r.capture_time is not None), key=lambda r: (r.capture_time, r.source.name))
        untimed = sorted((r for r in flat if r.capture_time is None), key=lambda r: r.source.name)
        n, previous = 0, None
        for r in timed:
            if previous is None or r.capture_time - previous > cfg.time_gap_s:
                n += 1
            r.group, previous = f"auto{n:03d}", r.capture_time
        for r in untimed:
            n += 1
            r.group = f"auto{n:03d}"


def _write_outputs(records: list[ImageRecord], cfg: CleanConfig) -> None:
    out_root = prepare_output_dir(cfg.output_dir)
    flagged_root = prepare_output_dir(Path(cfg.reports_dir) / "flagged")
    used: set[Path] = set()
    for r in records:
        if r.status != "kept":
            continue
        class_dir = out_root / r.class_name
        class_dir.mkdir(exist_ok=True)
        base = f"{r.group}{GROUP_SEPARATOR}{safe_name(r.source.stem, 'img')}"
        target, k = class_dir / f"{base}.jpg", 2
        while target in used:
            target, k = class_dir / f"{base}-{k}.jpg", k + 1
        used.add(target)
        target.write_bytes(r.jpeg)
        r.output = target
        for flag in r.flags:
            if flag in REVIEW_FLAGS:
                review_dir = flagged_root / flag
                review_dir.mkdir(exist_ok=True)
                (review_dir / f"{r.class_name}{GROUP_SEPARATOR}{target.name}").write_bytes(r.jpeg)
        r.jpeg = None  # free memory as we go


def _write_manifest(records: list[ImageRecord], cfg: CleanConfig, path: Path) -> None:
    columns = [
        "class", "group", "status", "reason", "flags", "source", "output", "width", "height",
        "blur", "brightness", "overexposed_frac", "capture_time", "sha256",
    ]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(columns)
        for r in records:
            writer.writerow([
                r.class_name,
                r.group,
                r.status,
                r.reason,
                ";".join(r.flags),
                _rel(r.source, cfg.input_dir),
                _rel(r.output, cfg.output_dir) if r.output else "",
                r.width or "",
                r.height or "",
                "" if np.isnan(r.blur) else f"{r.blur:.1f}",
                "" if np.isnan(r.brightness) else f"{r.brightness:.1f}",
                "" if np.isnan(r.overexposed) else f"{r.overexposed:.3f}",
                dt.datetime.fromtimestamp(r.capture_time).isoformat() if r.capture_time else "",
                r.sha256,
            ])


def _summarise(records: list[ImageRecord], warnings: list[str], cfg: CleanConfig) -> dict:
    per_class: dict[str, dict] = {}
    for class_name in cfg.class_names:
        recs = [r for r in records if r.class_name == class_name]
        kept = [r for r in recs if r.status == "kept"]
        group_sizes = Counter(r.group for r in kept)
        per_class[class_name] = {
            "found": len(recs),
            "kept": len(kept),
            "groups": len(group_sizes),
            "removed": dict(Counter(r.reason.split(":")[0].split(" ")[0] for r in recs if r.status != "kept")),
            "flagged": dict(Counter(f for r in kept for f in r.flags if f in REVIEW_FLAGS)),
        }
        if 0 < len(kept) < MIN_IMAGES_PER_CLASS:
            warnings.append(
                f"'{class_name}' has {len(kept)} images; the target is at least {MIN_IMAGES_PER_CLASS}."
            )
        if 0 < len(group_sizes) < 3:
            warnings.append(f"'{class_name}' has only {len(group_sizes)} sample group(s); the split needs 3+.")
        for group, size in sorted(group_sizes.items()):
            if size > MAX_SHOTS_PER_GROUP:
                warnings.append(
                    f"'{class_name}/{group}' has {size} images; check that the sub-folder holds one physical "
                    "sample (3-5 shots)."
                )
    counts = [c["kept"] for c in per_class.values() if c["kept"] > 0]
    if len(counts) > 1 and max(counts) / min(counts) > MAX_IMBALANCE_RATIO:
        warnings.append(
            f"Class imbalance: largest/smallest = {max(counts) / min(counts):.2f} "
            f"(keep it <= {MAX_IMBALANCE_RATIO}; training also applies class weights)."
        )
    settings = {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(cfg).items()}
    settings["class_names"] = list(cfg.class_names)
    return {
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "settings": settings,
        "totals": {
            "files_found": len(records),
            "kept": sum(c["kept"] for c in per_class.values()),
            "removed": sum(1 for r in records if r.status != "kept"),
            "flagged_for_review": sum(sum(c["flagged"].values()) for c in per_class.values()),
        },
        "per_class": per_class,
        "warnings": warnings,
    }


def clean_dataset(cfg: CleanConfig, log: Callable[[str], None] = print) -> dict:
    """Run the full cleaning step and return the summary dict (also written to clean_summary.json)."""
    register_heif()
    cfg.input_dir, cfg.output_dir, cfg.reports_dir = map(Path, (cfg.input_dir, cfg.output_dir, cfg.reports_dir))
    records, warnings = discover(cfg.input_dir, cfg.class_names)
    candidates = [r for r in records if r.status == "kept"]
    log(f"Found {len(records)} files ({len(candidates)} with an image extension) in {cfg.input_dir}")

    tasks = [(str(r.source), cfg.out_size, cfg.min_side, cfg.metric_side, cfg.jpeg_quality) for r in candidates]
    for r, res in zip(candidates, _run_parallel(tasks, cfg.workers, log)):
        for key in ("sha256", "width", "height", "capture_time", "blur", "brightness", "overexposed",
                    "thumb", "jpeg"):
            if key in res:
                setattr(r, key, res[key])
        r.flags.extend(res.get("flags", []))
        if res["status"] != "kept":
            r.remove(res["reason"])

    kept = [r for r in records if r.status == "kept"]
    _flag_quality(kept, cfg)
    _assign_groups(records, cfg)
    _remove_duplicates(records, cfg)
    _write_outputs(records, cfg)

    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    _write_manifest(records, cfg, cfg.reports_dir / "clean_manifest.csv")
    summary = _summarise(records, warnings, cfg)
    (cfg.reports_dir / "clean_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def format_summary(summary: dict) -> str:
    """Human-readable table of a clean_dataset() summary."""
    lines = [f"{'class':<16}{'found':>7}{'kept':>7}{'groups':>8}  removed / flagged", "-" * 72]
    for name, c in summary["per_class"].items():
        removed = ", ".join(f"{k}={v}" for k, v in sorted(c["removed"].items())) or "-"
        flagged = ", ".join(f"{k}={v}" for k, v in sorted(c["flagged"].items())) or "-"
        lines.append(f"{name:<16}{c['found']:>7}{c['kept']:>7}{c['groups']:>8}  {removed} / {flagged}")
    t = summary["totals"]
    lines += ["-" * 72, f"{'total':<16}{t['files_found']:>7}{t['kept']:>7}"
              f"{'':>8}  removed={t['removed']}, flagged for review={t['flagged_for_review']}"]
    if summary["warnings"]:
        lines += ["", "Warnings:"] + [f"  ! {w}" for w in summary["warnings"]]
    return "\n".join(lines)
