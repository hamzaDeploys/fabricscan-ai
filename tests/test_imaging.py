import datetime as dt
import io

import numpy as np
from PIL import Image, ImageFilter

from fabricscan.imaging import (
    center_crop_square,
    encode_jpeg,
    gray_array,
    laplacian_variance,
    read_capture_time,
    rmse_matrix,
    thumbnail,
    to_rgb,
)
from fabricscan.synthetic import generate_sample


def _sample(seed: int, cls: str = "hole", size: int = 256) -> Image.Image:
    arr = generate_sample(np.random.default_rng(seed), cls, size)
    return Image.fromarray(arr.astype(np.uint8))


def test_center_crop_square_keeps_the_centre():
    img = Image.new("RGB", (400, 300), "white")
    img.putpixel((200, 150), (255, 0, 0))
    crop = center_crop_square(img)
    assert crop.size == (300, 300)
    assert crop.getpixel((150, 150)) == (255, 0, 0)


def test_to_rgb_composites_transparency_onto_white():
    rgba = Image.new("RGBA", (4, 4), (0, 0, 0, 0))
    assert to_rgb(rgba).getpixel((0, 0)) == (255, 255, 255)
    assert to_rgb(Image.new("L", (2, 2), 7)).getpixel((0, 0)) == (7, 7, 7)


def test_blur_lowers_laplacian_variance():
    img = _sample(0)
    sharp = laplacian_variance(gray_array(img))
    blurred = laplacian_variance(gray_array(img.filter(ImageFilter.GaussianBlur(4))))
    assert blurred < sharp / 10


def test_rmse_separates_recompressed_copies_from_other_images():
    original = _sample(1)
    with Image.open(io.BytesIO(encode_jpeg(original.resize((230, 230)), quality=70))) as copy:
        copy = copy.resize((256, 256)).copy()
    other = _sample(2)
    m = rmse_matrix(np.stack([thumbnail(x) for x in (original, copy, other)]))
    assert np.allclose(m, m.T)
    assert np.allclose(np.diag(m), 0, atol=1e-3)
    assert m[0, 1] < 3.0 < m[0, 2]


def test_capture_time_is_read_from_exif_and_missing_is_none():
    exif = Image.Exif()
    exif[0x0132] = "2026:10:28 10:15:30"
    buf = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buf, "JPEG", exif=exif.tobytes())
    with Image.open(buf) as img:
        assert read_capture_time(img) == dt.datetime(2026, 10, 28, 10, 15, 30).timestamp()
    assert read_capture_time(Image.new("RGB", (8, 8))) is None


def test_encode_jpeg_drops_metadata():
    exif = Image.Exif()
    exif[0x0132] = "2026:10:28 10:15:30"
    buf = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buf, "JPEG", exif=exif.tobytes())
    with Image.open(buf) as img:
        clean = encode_jpeg(img)
    with Image.open(io.BytesIO(clean)) as img:
        assert not img.getexif()
