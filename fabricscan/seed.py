"""One call to make a run reproducible."""

from __future__ import annotations

import os
import random

import numpy as np

from .config import SEED


def set_global_seed(seed: int = SEED, deterministic_ops: bool = False) -> None:
    """Seed Python, NumPy and TensorFlow/Keras.

    ``deterministic_ops=True`` additionally forces deterministic TensorFlow kernels (bit-identical reruns on
    the same hardware, at some speed cost). Note: PYTHONHASHSEED only affects interpreters started after this
    call, e.g. tf.data worker processes.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import keras
        import tensorflow as tf
    except ImportError:
        return
    keras.utils.set_random_seed(seed)
    if deterministic_ops:
        tf.config.experimental.enable_op_determinism()
