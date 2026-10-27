"""tf.data input pipeline: loading, preprocessing and training-time augmentation.

Design rules (explained in docs/PREPROCESSING.md):

* The pipeline yields float32 RGB images with values in [0, 255] and size IMG_SIZE x IMG_SIZE.
  Pixel normalisation lives INSIDE each model (a Rescaling layer), so the browser can feed raw pixels to
  the exported ONNX model and there is a single source of truth for preprocessing.
* Augmentation runs only on the training split and only here in tf.data, never inside the model, so the
  exported graph stays small and contains no random ops.
* Every random op is stateless and driven by a seeded random stream, so runs are reproducible.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import tensorflow as tf

from .config import CLASS_NAMES, IMG_SIZE, PROCESSED_SIZE, SEED

AUTOTUNE = tf.data.AUTOTUNE
_SUFFIXES = (".jpg", ".jpeg", ".png")


@dataclass(frozen=True)
class AugmentConfig:
    """Training-time augmentation parameters (see docs/PREPROCESSING.md for the reasoning)."""

    max_rotation_deg: float = 15.0  # small random rotation, reflect-padded
    min_crop_scale: float = 0.80  # random zoom: crop 80-100 % of the side, resize back (up to 1.25x zoom)
    flips: bool = True  # horizontal + vertical flips
    rot90: bool = True  # random multiples of 90 degrees: fabric has no "up"
    brightness: float = 0.20  # multiply by U(0.8, 1.2)
    contrast: float = 0.15  # contrast factor U(0.85, 1.15)
    max_noise_std: float = 8.0  # Gaussian sensor noise, std ~ U(0, 8) on the 0-255 scale


def list_split(split_dir: Path, class_names: Sequence[str] = CLASS_NAMES) -> tuple[list[str], list[int]]:
    """File paths and integer labels of <split_dir>/<class>/*.jpg, in class-index order."""
    paths: list[str] = []
    labels: list[int] = []
    for index, name in enumerate(class_names):
        class_dir = Path(split_dir) / name
        files = sorted(p for p in class_dir.glob("*") if p.suffix.lower() in _SUFFIXES) if class_dir.is_dir() else []
        paths += [str(p) for p in files]
        labels += [index] * len(files)
    if not paths:
        raise FileNotFoundError(f"No images found under {split_dir}/<class>/")
    return paths, labels


def class_weights(labels: Sequence[int], num_classes: int = len(CLASS_NAMES)) -> dict[int, float]:
    """'Balanced' weights n_total / (n_classes * n_c), so every class contributes equally to the loss."""
    counts = np.bincount(np.asarray(labels), minlength=num_classes)
    total = counts.sum()
    return {i: float(total / (num_classes * c)) for i, c in enumerate(counts) if c > 0}


def load_image(path: tf.Tensor, load_size: int = PROCESSED_SIZE) -> tf.Tensor:
    """Read and decode a file into a uint8 (load_size, load_size, 3) tensor."""
    img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, [load_size, load_size], antialias=True)
    return tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)


def resize_for_model(image: tf.Tensor, img_size: int = IMG_SIZE) -> tf.Tensor:
    """Evaluation/inference path: the whole square image resized to the model input. No randomness."""
    image = tf.image.resize(tf.cast(image, tf.float32), [img_size, img_size], antialias=True)
    return tf.clip_by_value(image, 0.0, 255.0)


def _rotate(image: tf.Tensor, angle: tf.Tensor) -> tf.Tensor:
    """Rotate a HxWxC float image by `angle` radians around its centre, reflecting at the borders."""
    h = tf.cast(tf.shape(image)[0], tf.float32)
    w = tf.cast(tf.shape(image)[1], tf.float32)
    cos, sin = tf.cos(angle), tf.sin(angle)
    x_offset = ((w - 1) - (cos * (w - 1) - sin * (h - 1))) / 2.0
    y_offset = ((h - 1) - (sin * (w - 1) + cos * (h - 1))) / 2.0
    transform = tf.stack([cos, -sin, x_offset, sin, cos, y_offset, 0.0, 0.0])[None, :]
    out = tf.raw_ops.ImageProjectiveTransformV3(
        images=image[None],
        transforms=transform,
        output_shape=tf.shape(image)[:2],
        fill_value=0.0,
        interpolation="BILINEAR",
        fill_mode="REFLECT",
    )
    return out[0]


def augment(image: tf.Tensor, seed: tf.Tensor, img_size: int = IMG_SIZE,
            cfg: AugmentConfig = AugmentConfig()) -> tf.Tensor:
    """Random training augmentation. `seed` is a shape-[2] integer tensor; same seed -> same result.

    Order: rotation -> zoom (crop + resize) -> flips / 90-degree turns -> brightness -> contrast -> noise.
    Hue and saturation are deliberately NOT changed: colour is the evidence for stains.
    """
    image = tf.cast(image, tf.float32)
    seeds = tf.random.experimental.stateless_split(tf.cast(seed, tf.int64), num=10)

    if cfg.max_rotation_deg > 0:
        max_rad = cfg.max_rotation_deg * math.pi / 180.0
        image = _rotate(image, tf.random.stateless_uniform([], seeds[0], -max_rad, max_rad))

    side = tf.shape(image)[0]
    scale = tf.random.stateless_uniform([], seeds[1], cfg.min_crop_scale, 1.0)
    crop = tf.cast(tf.round(tf.cast(side, tf.float32) * scale), tf.int32)
    image = tf.image.stateless_random_crop(image, tf.stack([crop, crop, 3]), seed=seeds[2])
    image = tf.image.resize(image, [img_size, img_size], antialias=True)

    if cfg.flips:
        image = tf.image.stateless_random_flip_left_right(image, seeds[3])
        image = tf.image.stateless_random_flip_up_down(image, seeds[4])
    if cfg.rot90:
        image = tf.image.rot90(image, tf.random.stateless_uniform([], seeds[5], 0, 4, dtype=tf.int32))

    image = image * tf.random.stateless_uniform([], seeds[6], 1.0 - cfg.brightness, 1.0 + cfg.brightness)
    factor = tf.random.stateless_uniform([], seeds[7], 1.0 - cfg.contrast, 1.0 + cfg.contrast)
    mean = tf.reduce_mean(image, axis=[0, 1], keepdims=True)
    image = (image - mean) * factor + mean

    if cfg.max_noise_std > 0:
        std = tf.random.stateless_uniform([], seeds[8], 0.0, cfg.max_noise_std)
        image = image + tf.random.stateless_normal(tf.shape(image), seeds[9]) * std

    image = tf.clip_by_value(image, 0.0, 255.0)
    return tf.ensure_shape(image, [img_size, img_size, 3])


def make_dataset(
    split_dir: Path,
    *,
    training: bool,
    batch_size: int = 32,
    img_size: int = IMG_SIZE,
    load_size: int = PROCESSED_SIZE,
    seed: int = SEED,
    augment_cfg: AugmentConfig = AugmentConfig(),
    cache: bool = True,
    class_names: Sequence[str] = CLASS_NAMES,
) -> tf.data.Dataset:
    """Batched (images, labels) dataset for one split.

    Training: shuffle every epoch + augmentation. Validation/test: deterministic resize only.
    Decoded uint8 images are cached in memory (about 200 MB per 1,000 images at 256 px).
    """
    paths, labels = list_split(split_dir, class_names)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(lambda p, y: (load_image(p, load_size), y), num_parallel_calls=AUTOTUNE)
    if cache:
        ds = ds.cache()
    if training:
        ds = ds.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)
        seeds = tf.data.Dataset.random(seed=seed, rerandomize_each_iteration=True).batch(2)
        ds = tf.data.Dataset.zip((ds, seeds)).map(
            lambda xy, s: (augment(xy[0], s, img_size, augment_cfg), xy[1]),
            num_parallel_calls=AUTOTUNE,
        )
    else:
        ds = ds.map(lambda x, y: (resize_for_model(x, img_size), y), num_parallel_calls=AUTOTUNE)
    return ds.batch(batch_size).prefetch(AUTOTUNE)
