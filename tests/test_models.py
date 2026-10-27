import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
pytest.importorskip("tf2onnx")
pytest.importorskip("onnxruntime")

from fabricscan.config import NUM_CLASSES  # noqa: E402
from fabricscan.export import cam_function, export_onnx, verify_onnx  # noqa: E402
from fabricscan.models import build_scratch_cnn, build_transfer_model, unfreeze_top  # noqa: E402

X = np.random.default_rng(0).uniform(0, 255, (2, 224, 224, 3)).astype(np.float32)


@pytest.mark.parametrize("pooling", ["max", "avg"])
def test_scratch_cnn_shapes_and_cam(pooling):
    model = build_scratch_cnn(pooling=pooling)
    out = cam_function(model)(tf.constant(X))
    assert out["probs"].shape == (2, NUM_CLASSES)
    assert out["cam"].shape == (2, 14, 14, NUM_CLASSES)
    np.testing.assert_allclose(out["probs"].numpy().sum(axis=1), 1.0, rtol=1e-5)


def test_cam_equals_grad_cam_for_max_pooling():
    """With a global-pool -> Dense head, Grad-CAM (channel-averaged gradients) is proportional to the exported CAM."""
    model = build_scratch_cnn(pooling="max")
    features_model = tf.keras.Model(model.inputs, [model.get_layer("probs").input, model.get_layer("pool").input])
    head = model.get_layer("probs")
    x = tf.constant(X[:1])
    with tf.GradientTape() as tape:
        pooled, feats = features_model(x, training=False)
        tape.watch(feats)
        logits = tf.matmul(tf.reduce_max(feats, axis=[1, 2]), head.kernel) + head.bias
        score = logits[0, 2]
    grads = tape.gradient(score, feats)
    weights = tf.reduce_mean(grads, axis=[1, 2])  # Grad-CAM channel weights
    grad_cam = tf.einsum("bhwc,bc->bhw", feats, weights).numpy()[0]
    cam = cam_function(model)(x)["cam"].numpy()[0, :, :, 2]
    ratio = grad_cam / np.where(np.abs(cam) > 1e-6, cam, np.nan)
    assert np.nanstd(ratio) / abs(np.nanmean(ratio)) < 1e-3  # same map up to one positive factor


def test_onnx_export_matches_keras(tmp_path):
    model = build_scratch_cnn()
    path = export_onnx(model, tmp_path / "m.onnx")
    check = verify_onnx(model, path, X)
    assert check["ok"], check
    assert check["cam_shape"] == [14, 14, NUM_CLASSES]


def test_transfer_model_unfreezes_only_top_layers():
    model, backbone = build_transfer_model(weights=None)
    assert not backbone.trainable
    n = unfreeze_top(backbone, 40)
    assert 0 < n <= 40
    frozen_bn = [l for l in backbone.layers if isinstance(l, tf.keras.layers.BatchNormalization) and l.trainable]
    assert frozen_bn == []
    assert model.get_layer("pool").__class__.__name__ == "GlobalAveragePooling2D"
