"""The two model architectures compared in this project.

1. ``build_scratch_cnn``: a custom VGG-style CNN with BatchNorm, Dropout and L2, trained from random weights.
2. ``build_transfer_model``: MobileNetV2 pre-trained on ImageNet, first frozen, then partly fine-tuned.

Both end in the same kind of head: global pooling ("pool", average or max) -> Dropout -> Dense softmax ("probs").
For either pooling, the channel-averaged gradient of class score k w.r.t. feature map A_c is w[c, k] / Z, so Grad-CAM
reduces to ReLU(sum_c w[c, k] * A[:, :, c]) up to a positive factor (Class Activation Mapping). The exported model
therefore returns a defect heatmap with every prediction (see fabricscan/export.py).

Max pooling suits defect detection: a defect covers only a few of the 14x14 feature positions, and averaging
over all of them dilutes its signal, which made the scratch CNN underfit in early experiments.

Pixel normalisation is the first layer of each model, so both take raw RGB values in [0, 255].
"""

from __future__ import annotations

import keras
from keras import layers, regularizers

from .config import IMG_SIZE, NUM_CLASSES


def _global_pool(pooling: str):
    if pooling == "max":
        return layers.GlobalMaxPooling2D(name="pool")
    if pooling == "avg":
        return layers.GlobalAveragePooling2D(name="pool")
    raise ValueError(f"pooling must be 'max' or 'avg', got {pooling!r}")


def _conv_bn_relu(x, filters: int, name: str, stride: int = 1, l2: float = 1e-4):
    """3x3 convolution (no bias: BatchNorm has its own shift) -> BatchNorm -> ReLU."""
    x = layers.Conv2D(
        filters, 3, strides=stride, padding="same", use_bias=False, kernel_initializer="he_normal",
        kernel_regularizer=regularizers.l2(l2), name=f"{name}_conv",
    )(x)
    x = layers.BatchNormalization(name=f"{name}_bn")(x)
    return layers.ReLU(name=f"{name}_relu")(x)


def build_scratch_cnn(num_classes: int = NUM_CLASSES, img_size: int = IMG_SIZE, dropout: float = 0.3,
                      l2: float = 1e-5, spatial_dropout: float = 0.05, pooling: str = "max") -> keras.Model:
    """Custom CNN: 4 convolutional stages (32-64-128-256 filters), ~1.2 M parameters, 14x14 CAM resolution.

    224 -> stem (stride 2) 112 -> pool 56 -> pool 28 -> pool 14 -> last conv block at 14x14 -> global pool -> softmax.
    The stride-2 stem keeps the compute low enough to train on a laptop CPU, and the last stage is not pooled so
    the heatmap keeps a 14x14 grid (twice as fine as MobileNetV2's 7x7).
    """
    inputs = keras.Input((img_size, img_size, 3), name="image")
    x = layers.Rescaling(1.0 / 255, name="rescale")(inputs)

    x = _conv_bn_relu(x, 32, "stem", stride=2, l2=l2)
    x = _conv_bn_relu(x, 32, "s1c1", l2=l2)
    x = layers.MaxPooling2D(name="s1_pool")(x)

    x = _conv_bn_relu(x, 64, "s2c1", l2=l2)
    x = _conv_bn_relu(x, 64, "s2c2", l2=l2)
    x = layers.MaxPooling2D(name="s2_pool")(x)
    x = layers.SpatialDropout2D(spatial_dropout, name="s2_drop")(x)

    x = _conv_bn_relu(x, 128, "s3c1", l2=l2)
    x = _conv_bn_relu(x, 128, "s3c2", l2=l2)
    x = layers.MaxPooling2D(name="s3_pool")(x)
    x = layers.SpatialDropout2D(spatial_dropout, name="s3_drop")(x)

    x = _conv_bn_relu(x, 256, "s4c1", l2=l2)
    x = _conv_bn_relu(x, 256, "s4c2", l2=l2)  # feature maps used for the heatmap

    x = _global_pool(pooling)(x)
    x = layers.Dropout(dropout, name="head_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", kernel_regularizer=regularizers.l2(l2),
                           name="probs")(x)
    return keras.Model(inputs, outputs, name="fabricscan_scratch_cnn")


def build_transfer_model(num_classes: int = NUM_CLASSES, img_size: int = IMG_SIZE, dropout: float = 0.3,
                         weights: str | None = "imagenet", pooling: str = "avg") -> tuple[keras.Model, keras.Model]:
    """MobileNetV2 backbone (frozen) + the shared head. Returns (model, backbone) so the backbone can be unfrozen.

    The backbone is always called with training=False, so its BatchNorm layers keep their ImageNet statistics,
    even during fine-tuning. On small datasets, updating those statistics usually destroys the pre-trained
    features.
    """
    inputs = keras.Input((img_size, img_size, 3), name="image")
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0, name="rescale")(inputs)  # MobileNetV2 expects [-1, 1]
    backbone = keras.applications.MobileNetV2(include_top=False, weights=weights,
                                              input_shape=(img_size, img_size, 3))
    backbone.trainable = False
    x = backbone(x, training=False)
    x = _global_pool(pooling)(x)
    x = layers.Dropout(dropout, name="head_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="probs")(x)
    return keras.Model(inputs, outputs, name="fabricscan_mobilenetv2"), backbone


def unfreeze_top(backbone: keras.Model, n_layers: int) -> int:
    """Make the last ``n_layers`` of the backbone trainable (BatchNorm layers stay frozen). Returns how many."""
    backbone.trainable = True
    trainable = 0
    for i, layer in enumerate(backbone.layers):
        keep_frozen = i < len(backbone.layers) - n_layers or isinstance(layer, layers.BatchNormalization)
        layer.trainable = not keep_frozen
        trainable += 0 if keep_frozen else 1
    return trainable
