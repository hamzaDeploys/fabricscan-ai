"""Training loops for both models (shared by scripts/train_models.py and the notebooks)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import keras
import tensorflow as tf

from .models import build_scratch_cnn, build_transfer_model, unfreeze_top


@dataclass
class TrainResult:
    model: keras.Model
    history: dict[str, list[float]] = field(default_factory=dict)
    train_time_s: float = 0.0
    fine_tune_start_epoch: int | None = None  # transfer model: index of the first fine-tuning epoch

    @property
    def epochs(self) -> int:
        return len(self.history.get("loss", []))


def compile_model(model: keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )


def _callbacks(log_path: Path | None, patience: int) -> list[keras.callbacks.Callback]:
    callbacks: list[keras.callbacks.Callback] = [
        # Stop when validation loss stops improving and roll back to the best epoch (fights overfitting).
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
        # Halve the learning rate when progress stalls (helps the loss settle into a better minimum).
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=max(2, patience // 3),
                                          min_lr=1e-6),
    ]
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        callbacks.append(keras.callbacks.CSVLogger(str(log_path), append=log_path.exists()))
    return callbacks


def _merge(history: dict[str, list[float]], new: keras.callbacks.History) -> None:
    for key, values in new.history.items():
        if key == "learning_rate":
            key = "lr"
        history.setdefault(key, []).extend(float(v) for v in values)


def train_scratch(train_ds: tf.data.Dataset, val_ds: tf.data.Dataset, class_weight: dict[int, float] | None, *,
                  epochs: int = 60, learning_rate: float = 1e-3, patience: int = 10, log_dir: Path | None = None,
                  verbose: int = 2, model_kwargs: dict | None = None) -> TrainResult:
    """Train the custom CNN from random weights."""
    model = build_scratch_cnn(**(model_kwargs or {}))
    compile_model(model, learning_rate)
    log = Path(log_dir) / "scratch_cnn_log.csv" if log_dir else None
    if log and log.exists():
        log.unlink()
    start = time.perf_counter()
    hist = model.fit(train_ds, validation_data=val_ds, epochs=epochs, class_weight=class_weight,
                     callbacks=_callbacks(log, patience), verbose=verbose)
    result = TrainResult(model, train_time_s=time.perf_counter() - start)
    _merge(result.history, hist)
    return result


def train_transfer(train_ds: tf.data.Dataset, val_ds: tf.data.Dataset, class_weight: dict[int, float] | None, *,
                   epochs_frozen: int = 12, epochs_finetune: int = 20, lr_frozen: float = 1e-3,
                   lr_finetune: float = 1e-5, unfreeze_layers: int = 40, patience: int = 6,
                   log_dir: Path | None = None, verbose: int = 2, weights: str | None = "imagenet",
                   model_kwargs: dict | None = None) -> TrainResult:
    """Two-stage transfer learning.

    Stage 1: backbone frozen, only the new head learns (high learning rate, fast).
    Stage 2: the top ``unfreeze_layers`` backbone layers are unfrozen and trained with a 100x smaller learning
    rate, so the pre-trained features adapt to fabric textures without being destroyed.
    """
    model, backbone = build_transfer_model(weights=weights, **(model_kwargs or {}))
    log = Path(log_dir) / "mobilenetv2_log.csv" if log_dir else None
    if log and log.exists():
        log.unlink()
    history: dict[str, list[float]] = {}
    start = time.perf_counter()

    compile_model(model, lr_frozen)
    hist = model.fit(train_ds, validation_data=val_ds, epochs=epochs_frozen, class_weight=class_weight,
                     callbacks=_callbacks(log, patience), verbose=verbose)
    _merge(history, hist)
    fine_tune_start = len(history["loss"])

    unfreeze_top(backbone, unfreeze_layers)
    compile_model(model, lr_finetune)  # re-compile so the new trainable flags take effect
    hist = model.fit(train_ds, validation_data=val_ds, initial_epoch=fine_tune_start,
                     epochs=fine_tune_start + epochs_finetune, class_weight=class_weight,
                     callbacks=_callbacks(log, patience), verbose=verbose)
    _merge(history, hist)
    return TrainResult(model, history, time.perf_counter() - start, fine_tune_start)
