"""Train -> evaluate -> export -> publish-to-website, as one reusable function.

Used by ``scripts/train_models.py`` (real dataset) and ``scripts/build_demo_model.py`` (synthetic demo).
The notebooks run the same steps one cell at a time, with plots and explanations.
"""

from __future__ import annotations

import base64
import datetime as dt
import json
import random
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from . import config
from .splitting import group_of, group_sources

def publish_web_models(web_models: Path, models_dir: Path, entries: dict, best: str, demo: bool, created: str) -> None:
    """The models the website's dropdown offers (models.json): the best first, as fabricscan.onnx + model_meta.json."""
    from .export import write_model_meta

    choices = []
    for key in sorted(entries, key=lambda k: k != best):
        file = MODEL_INFO[key]["file"]
        onnx_name, meta_name = ("fabricscan.onnx", "model_meta.json") if key == best else (f"{file}.onnx", f"{file}.json")
        shutil.copyfile(models_dir / f"{file}.onnx", web_models / onnx_name)
        write_model_meta(web_models / meta_name, model_key=key, architecture=entries[key]["name"],
                         cam_shape=entries[key]["onnx_check"]["cam_shape"], demo=demo,
                         test_accuracy=entries[key]["test"]["accuracy"], created=created, model_file=onnx_name)
        choices.append({"key": key, "name": entries[key]["name"], "meta": meta_name})
    (web_models / "models.json").write_text(json.dumps(choices, indent=2), encoding="utf-8")


MODEL_INFO = {
    "scratch": {
        "name": "Custom CNN (from scratch)",
        "file": "scratch_cnn",
        "architecture": "4-stage VGG-style CNN with BatchNorm, Dropout and L2, trained from random weights",
    },
    "transfer": {
        "name": "MobileNetV2 (transfer learning)",
        "file": "mobilenetv2",
        "architecture": "ImageNet MobileNetV2, frozen then top 40 layers fine-tuned",
    },
}


@dataclass
class TrainingPlan:
    epochs_scratch: int = 60
    epochs_frozen: int = 12
    epochs_finetune: int = 20
    patience_scratch: int = 10
    patience_transfer: int = 6
    batch_size: int = 32
    seed: int = config.SEED


def _round_history(history: dict[str, list[float]]) -> dict[str, list[float]]:
    return {k: [round(v, 5) for v in vals] for k, vals in history.items()}


def _pick_best(entries: dict[str, dict]) -> str:
    return max(entries, key=lambda k: (round(entries[k]["test"]["f1_macro"], 4),
                                       round(entries[k]["test"]["accuracy"], 4), -entries[k]["onnx_size_mb"]))


def _copy_samples(test_dir: Path, samples_dir: Path, per_class: int, seed: int, demo: bool) -> list[dict]:
    """Copy a few test images per class (from different physical samples) for the website's sample strip."""
    if samples_dir.exists():
        shutil.rmtree(samples_dir)
    samples_dir.mkdir(parents=True)
    rng = random.Random(seed)
    sources = group_sources(config.DATASET_DIR / "SOURCES.csv")
    manifest = []
    for class_name in config.CLASS_NAMES:
        by_group: dict[str, list[Path]] = defaultdict(list)
        for f in sorted((test_dir / class_name).glob("*.jpg")):
            by_group[group_of(f)].append(f)
        groups = sorted(by_group)
        rng.shuffle(groups)
        for i, g in enumerate(groups[:per_class], 1):
            src = rng.choice(by_group[g])
            name = f"{class_name}-{i}.jpg"
            shutil.copyfile(src, samples_dir / name)
            if demo:
                source = "synthetic test image (demo)"
            else:
                source = f"{sources.get(g, 'Own photo')}, test split"
            manifest.append({"file": f"samples/{name}", "class": class_name, "label": config.DISPLAY_NAMES[class_name],
                             "source": source})
    (samples_dir / "samples.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _write_parity_fixture(onnx_path: Path, image_path: Path, target: Path) -> None:
    """One preprocessed image + ONNX Runtime's output, so a web unit test can check the browser engine matches."""
    import onnxruntime as ort
    import tensorflow as tf

    from .data import load_image, resize_for_model

    pixels = tf.cast(tf.round(resize_for_model(load_image(tf.constant(str(image_path))))), tf.uint8).numpy()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    out = dict(zip([o.name for o in session.get_outputs()],
                   session.run(None, {"image": pixels[None].astype(np.float32)})))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({
        "image_uint8_base64": base64.b64encode(pixels.tobytes()).decode(),
        "shape": list(pixels.shape),
        "probs": out["probs"][0].tolist(),
        "cam_shape": list(out["cam"].shape[1:]),
    }), encoding="utf-8")


def run_training(
    splits_dir: Path,
    *,
    models: Sequence[str] = ("scratch", "transfer"),
    plan: TrainingPlan = TrainingPlan(),
    models_dir: Path = config.MODELS_DIR,
    results_dir: Path = config.RESULTS_DIR / "metrics",
    web_public: Path | None = config.PROJECT_ROOT / "web" / "public",
    dataset_summary: Path | None = config.RESULTS_DIR / "dataset" / "dataset_summary.json",
    demo: bool = False,
    verbose: int = 2,
    log: Callable[[str], None] = print,
) -> dict:
    """Train the requested models, evaluate on the test split, export ONNX and (optionally) publish to web/."""
    import keras

    from .data import class_weights, list_split, make_dataset
    from .evaluation import compute_metrics, file_size_mb, keras_latency_ms, onnx_latency_ms, predict_dataset
    from .export import export_onnx, verify_onnx
    from .seed import set_global_seed
    from .training import train_scratch, train_transfer

    splits_dir, models_dir, results_dir = Path(splits_dir), Path(models_dir), Path(results_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    set_global_seed(plan.seed)

    train_ds = make_dataset(splits_dir / "train", training=True, batch_size=plan.batch_size, seed=plan.seed)
    val_ds = make_dataset(splits_dir / "val", training=False, batch_size=plan.batch_size)
    test_ds = make_dataset(splits_dir / "test", training=False, batch_size=plan.batch_size)
    _, train_labels = list_split(splits_dir / "train")
    weights = class_weights(train_labels)
    counts = {s: len(list_split(splits_dir / s)[0]) for s in config.SPLITS}
    log(f"Images: {counts}. Class weights: { {k: round(v, 2) for k, v in weights.items()} }")
    verify_batch = next(iter(test_ds))[0].numpy()[:8]

    entries: dict[str, dict] = {}
    for key in models:
        info = MODEL_INFO[key]
        log(f"\n=== Training {info['name']} ===")
        keras.backend.clear_session()
        set_global_seed(plan.seed)
        if key == "scratch":
            result = train_scratch(train_ds, val_ds, weights, epochs=plan.epochs_scratch,
                                   patience=plan.patience_scratch, log_dir=results_dir, verbose=verbose)
        else:
            result = train_transfer(train_ds, val_ds, weights, epochs_frozen=plan.epochs_frozen,
                                    epochs_finetune=plan.epochs_finetune, patience=plan.patience_transfer,
                                    log_dir=results_dir, verbose=verbose)
        model = result.model
        keras_path = models_dir / f"{info['file']}.keras"
        model.save(keras_path)
        y_true, probs = predict_dataset(model, test_ds)
        metrics = compute_metrics(y_true, probs)
        onnx_path = export_onnx(model, models_dir / f"{info['file']}.onnx")
        check = verify_onnx(model, onnx_path, verify_batch)
        if not check["ok"]:
            raise RuntimeError(f"ONNX export of {key} does not match Keras: {check}")
        entries[key] = {
            "key": key,
            "name": info["name"],
            "architecture": info["architecture"],
            "params": int(model.count_params()),
            "keras_size_mb": round(file_size_mb(keras_path), 2),
            "onnx_size_mb": round(file_size_mb(onnx_path), 2),
            "keras_latency_ms": round(keras_latency_ms(model), 2),
            "onnx_latency_ms": round(onnx_latency_ms(onnx_path), 2),
            "epochs": result.epochs,
            "train_time_s": round(result.train_time_s, 1),
            "fine_tune_start_epoch": result.fine_tune_start_epoch,
            "best_val_accuracy": round(max(result.history.get("val_accuracy", [0.0])), 4),
            "history": _round_history(result.history),
            "test": metrics,
            "onnx_check": check,
        }
        log(f"{info['name']}: test accuracy {metrics['accuracy']:.3f}, macro F1 {metrics['f1_macro']:.3f}, "
            f"{entries[key]['epochs']} epochs in {result.train_time_s:.0f}s, ONNX {entries[key]['onnx_size_mb']} MB, "
            f"{entries[key]['onnx_latency_ms']} ms/image")

    best = _pick_best(entries)
    created = dt.datetime.now().isoformat(timespec="seconds")
    summary = {
        "demo": demo,
        "generated": created,
        "class_names": list(config.CLASS_NAMES),
        "display_names": config.DISPLAY_NAMES,
        "images": counts,
        "best_model": best,
        "selection_rule": "highest macro F1 on the test split (ties: accuracy, then smaller model)",
        "models": entries,
    }
    (results_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"\nBest model: {entries[best]['name']}. Metrics: {results_dir / 'metrics.json'}")

    if web_public is not None:
        web_public = Path(web_public)
        (web_public / "models").mkdir(parents=True, exist_ok=True)
        (web_public / "data").mkdir(parents=True, exist_ok=True)
        best_onnx = models_dir / f"{MODEL_INFO[best]['file']}.onnx"
        publish_web_models(web_public / "models", models_dir, entries, best, demo, created)
        shutil.copyfile(results_dir / "metrics.json", web_public / "data" / "metrics.json")
        if dataset_summary is not None and Path(dataset_summary).is_file():
            shutil.copyfile(dataset_summary, web_public / "data" / "dataset_summary.json")
        samples = _copy_samples(splits_dir / "test", web_public / "samples", 2, plan.seed, demo)
        _write_parity_fixture(best_onnx, web_public / samples[0]["file"],
                              web_public.parent / "src" / "test" / "fixtures" / "parity.json")
        log(f"Website assets updated in {web_public} (model: {entries[best]['name']}, demo={demo})")
    return summary
