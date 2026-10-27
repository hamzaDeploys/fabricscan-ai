import numpy as np
import pytest
from PIL import Image

tf = pytest.importorskip("tensorflow")

from fabricscan.config import CLASS_NAMES  # noqa: E402
from fabricscan.data import augment, class_weights, list_split, make_dataset, resize_for_model  # noqa: E402


@pytest.fixture(scope="module")
def image():
    rng = np.random.default_rng(0)
    return tf.constant(rng.integers(0, 256, (256, 256, 3), dtype=np.uint8))


def test_augment_shape_dtype_and_range(image):
    out = augment(image, tf.constant([1, 2], tf.int64))
    assert out.shape == (224, 224, 3) and out.dtype == tf.float32
    assert 0.0 <= float(tf.reduce_min(out)) and float(tf.reduce_max(out)) <= 255.0


def test_augment_is_reproducible_per_seed(image):
    a = augment(image, tf.constant([1, 2], tf.int64)).numpy()
    b = augment(image, tf.constant([1, 2], tf.int64)).numpy()
    c = augment(image, tf.constant([1, 3], tf.int64)).numpy()
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_eval_resize_is_deterministic(image):
    out = resize_for_model(image)
    assert out.shape == (224, 224, 3)
    np.testing.assert_array_equal(out.numpy(), resize_for_model(image).numpy())


def test_class_weights_are_balanced():
    assert class_weights([0, 0, 1], num_classes=2) == {0: 0.75, 1: 1.5}


@pytest.fixture(scope="module")
def split_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("split")
    rng = np.random.default_rng(1)
    for c in CLASS_NAMES:
        (root / c).mkdir()
        for k in range(3):
            Image.fromarray(rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)).save(root / c / f"s00{k}__x.jpg")
    return root


def test_list_split_orders_labels_by_class_index(split_dir):
    paths, labels = list_split(split_dir)
    assert len(paths) == 3 * len(CLASS_NAMES)
    assert labels == sorted(labels) and set(labels) == set(range(len(CLASS_NAMES)))


def test_training_dataset_changes_between_epochs(split_dir):
    ds = make_dataset(split_dir, training=True, batch_size=3 * len(CLASS_NAMES))
    first = next(iter(ds))
    second = next(iter(ds))
    assert first[0].shape == (3 * len(CLASS_NAMES), 224, 224, 3)
    assert not np.array_equal(first[0].numpy(), second[0].numpy())


def test_eval_dataset_is_ordered_and_repeatable(split_dir):
    ds = make_dataset(split_dir, training=False, batch_size=3 * len(CLASS_NAMES))
    x1, y1 = next(iter(ds))
    x2, y2 = next(iter(ds))
    np.testing.assert_array_equal(x1.numpy(), x2.numpy())
    assert y1.numpy().tolist() == sorted(y1.numpy().tolist())
