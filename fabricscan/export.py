"""Export a trained Keras model to ONNX for in-browser inference with onnxruntime-web.

The exported graph has one input and two outputs:

* ``image``  float32 [batch, 224, 224, 3], raw RGB 0-255 (normalisation is inside the model)
* ``probs``  float32 [batch, num_classes], softmax probabilities
* ``cam``    float32 [batch, h, w, num_classes], class activation maps: sum_c A[:, :, c] * w[c, k]

Because both models end in global pooling -> Dense, ``cam`` is exactly the (unnormalised) Grad-CAM map of every
class (see fabricscan/models.py), so the website gets a heatmap from the same forward pass as the prediction.
"""

from __future__ import annotations

import json
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf

from .config import CLASS_NAMES, DISPLAY_NAMES, IMG_SIZE


def _feature_and_head(model: keras.Model) -> tuple[keras.Model, tf.Variable]:
    """A model returning (probs, last feature maps) plus the Dense kernel that maps features to classes."""
    pool, head = model.get_layer("pool"), model.get_layer("probs")
    return keras.Model(model.inputs, [head.output, pool.input]), head.kernel


def cam_function(model: keras.Model, img_size: int = IMG_SIZE) -> tf.types.experimental.ConcreteFunction:
    """tf.function(image) -> {"probs", "cam"}; also used to compute Grad-CAM figures in Python."""
    features_model, kernel = _feature_and_head(model)
    _, h, w, c = features_model.outputs[1].shape
    num_classes = kernel.shape[-1]
    spec = tf.TensorSpec((None, img_size, img_size, 3), tf.float32, name="image")

    @tf.function(input_signature=[spec])
    def serve(image):
        probs, feats = features_model(image, training=False)
        cam = tf.reshape(tf.matmul(tf.reshape(feats, [-1, c]), kernel), [-1, h, w, num_classes])
        return {"probs": probs, "cam": cam}

    return serve


def export_onnx(model: keras.Model, path: Path, opset: int = 17, img_size: int = IMG_SIZE) -> Path:
    import tf2onnx

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serve = cam_function(model, img_size)
    spec = (tf.TensorSpec((None, img_size, img_size, 3), tf.float32, name="image"),)
    tf2onnx.convert.from_function(serve, input_signature=spec, opset=opset, output_path=str(path))
    return path


def verify_onnx(model: keras.Model, onnx_path: Path, images: np.ndarray) -> dict:
    """Run the same images through Keras and ONNX Runtime and compare. ``images``: float32 [n, 224, 224, 3]."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    out = dict(zip([o.name for o in session.get_outputs()], session.run(None, {"image": images})))
    ref = cam_function(model)(tf.constant(images))
    probs_diff = float(np.abs(out["probs"] - ref["probs"].numpy()).max())
    cam_diff = float(np.abs(out["cam"] - ref["cam"].numpy()).max() / (np.abs(ref["cam"].numpy()).max() + 1e-9))
    agree = float((out["probs"].argmax(1) == ref["probs"].numpy().argmax(1)).mean())
    return {
        "max_abs_diff_probs": probs_diff,
        "max_rel_diff_cam": cam_diff,
        "argmax_agreement": agree,
        "cam_shape": list(out["cam"].shape[1:]),
        "ok": probs_diff < 1e-4 and agree == 1.0,
    }


def write_model_meta(path: Path, *, model_key: str, architecture: str, cam_shape: list[int], demo: bool,
                     test_accuracy: float | None, created: str, img_size: int = IMG_SIZE,
                     model_file: str = "fabricscan.onnx") -> None:
    """model_meta.json: everything the website needs to run the ONNX model correctly."""
    meta = {
        "model_file": model_file,
        "model_key": model_key,
        "architecture": architecture,
        "demo": demo,
        "created": created,
        "input": {"name": "image", "layout": "NHWC", "size": img_size, "range": [0, 255], "channels": "RGB"},
        "outputs": {"probs": "probs", "cam": "cam"},
        "cam_shape": cam_shape,
        "class_names": list(CLASS_NAMES),
        "display_names": DISPLAY_NAMES,
        "test_accuracy": test_accuracy,
    }
    Path(path).write_text(json.dumps(meta, indent=2), encoding="utf-8")
