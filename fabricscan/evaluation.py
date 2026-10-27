"""Test-set metrics, speed and size measurements used to compare the two models."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Sequence

import keras
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from .config import CLASS_NAMES, IMG_SIZE


def predict_dataset(model: keras.Model, ds: tf.data.Dataset) -> tuple[np.ndarray, np.ndarray]:
    """(true labels, predicted probabilities) for a batched, non-shuffled dataset."""
    labels, probs = [], []
    for images, y in ds:
        probs.append(model(images, training=False).numpy())
        labels.append(y.numpy())
    return np.concatenate(labels), np.concatenate(probs)


def compute_metrics(y_true: np.ndarray, probs: np.ndarray, class_names: Sequence[str] = CLASS_NAMES) -> dict:
    """Accuracy, macro/weighted precision-recall-F1, per-class scores and the confusion matrix."""
    k = len(class_names)
    y_pred = probs.argmax(axis=1)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=list(range(k)), zero_division=0)
    macro = precision_recall_fscore_support(y_true, y_pred, labels=list(range(k)), average="macro", zero_division=0)
    weighted = precision_recall_fscore_support(y_true, y_pred, labels=list(range(k)), average="weighted",
                                               zero_division=0)
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(macro[0]),
        "recall_macro": float(macro[1]),
        "f1_macro": float(macro[2]),
        "precision_weighted": float(weighted[0]),
        "recall_weighted": float(weighted[1]),
        "f1_weighted": float(weighted[2]),
        "per_class": {
            name: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
            for i, name in enumerate(class_names)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(k))).tolist(),
    }


def keras_latency_ms(model: keras.Model, img_size: int = IMG_SIZE, runs: int = 30, warmup: int = 5) -> float:
    """Median single-image inference time in TensorFlow on this machine."""
    x = tf.random.uniform((1, img_size, img_size, 3), 0, 255)
    fn = tf.function(lambda t: model(t, training=False))
    for _ in range(warmup):
        fn(x)
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        fn(x).numpy()
        times.append((time.perf_counter() - start) * 1000)
    return float(np.median(times))


def onnx_latency_ms(onnx_path: Path, img_size: int = IMG_SIZE, runs: int = 30, warmup: int = 5) -> float:
    """Median single-image inference time in ONNX Runtime (CPU), the engine family the website uses."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    x = np.random.default_rng(0).uniform(0, 255, (1, img_size, img_size, 3)).astype(np.float32)
    for _ in range(warmup):
        session.run(None, {name: x})
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        session.run(None, {name: x})
        times.append((time.perf_counter() - start) * 1000)
    return float(np.median(times))


def file_size_mb(path: Path) -> float:
    return Path(path).stat().st_size / 1e6
