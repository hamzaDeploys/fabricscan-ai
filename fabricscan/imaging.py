"""Image utilities for the cleaning and splitting steps (NumPy + Pillow only, no TensorFlow)."""

from __future__ import annotations

import datetime as dt
import io

import numpy as np
from PIL import Image

# Phones with 108-200 MP modes exceed Pillow's default decompression-bomb limit (~179 MP).
Image.MAX_IMAGE_PIXELS = 300_000_000

_EXIF_IFD = 0x8769
_TAG_DATETIME_ORIGINAL = 0x9003
_TAG_DATETIME = 0x0132
_TAG_ORIENTATION = 0x0112

_heif_available: bool | None = None


def register_heif() -> bool:
    """Enable HEIC/HEIF decoding when pillow-heif is installed. Returns True if available."""
    global _heif_available
    if _heif_available is None:
        try:
            from pillow_heif import register_heif_opener
        except ImportError:
            _heif_available = False
        else:
            register_heif_opener()
            _heif_available = True
    return _heif_available


def read_capture_time(img: Image.Image) -> float | None:
    """Return the EXIF capture time as a POSIX timestamp, or None when it is missing or malformed."""
    try:
        exif = img.getexif()
        raw = exif.get_ifd(_EXIF_IFD).get(_TAG_DATETIME_ORIGINAL) or exif.get(_TAG_DATETIME)
        if not raw:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode(errors="ignore")
        return dt.datetime.strptime(str(raw).strip("\x00 "), "%Y:%m:%d %H:%M:%S").timestamp()
    except Exception:  # malformed EXIF comes in many shapes; it must never stop the cleaning run
        return None


def read_orientation(img: Image.Image) -> int:
    """EXIF orientation tag (1-8); 1 when absent."""
    try:
        return int(img.getexif().get(_TAG_ORIENTATION, 1))
    except Exception:
        return 1


def to_rgb(img: Image.Image) -> Image.Image:
    """Convert any Pillow mode to RGB, compositing transparent pixels onto white."""
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(background, rgba).convert("RGB")
    return img if img.mode == "RGB" else img.convert("RGB")


def center_crop_square(img: Image.Image) -> Image.Image:
    """Crop the largest centred square."""
    w, h = img.size
    side = min(w, h)
    left, top = (w - side) // 2, (h - side) // 2
    return img.crop((left, top, left + side, top + side))


def gray_array(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("L"), dtype=np.float32)


def laplacian_variance(gray: np.ndarray) -> float:
    """Variance of the 4-neighbour Laplacian. Low values mean few sharp edges, i.e. blur."""
    g = gray.astype(np.float32, copy=False)
    lap = g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:] - 4.0 * g[1:-1, 1:-1]
    return float(lap.var())


def exposure_stats(gray: np.ndarray) -> tuple[float, float]:
    """(mean brightness 0-255, fraction of clipped highlights >= 250)."""
    return float(gray.mean()), float((gray >= 250).mean())


THUMB_SIZE = 32


def thumbnail(img: Image.Image, size: int = THUMB_SIZE) -> np.ndarray:
    """Area-averaged size x size RGB thumbnail, flattened to uint8. The basis of near-duplicate detection."""
    return np.asarray(img.convert("RGB").resize((size, size), Image.Resampling.BOX), dtype=np.uint8).ravel()


def rmse_matrix(thumbs: np.ndarray) -> np.ndarray:
    """Pairwise root-mean-square pixel difference (0-255 scale) between thumbnail rows: (n, d) -> (n, n).

    A re-saved, re-compressed or slightly resized copy of a photo scores about 1; a different shot of the
    same sample (moved camera or changed light) scores 10+.
    """
    x = thumbs.astype(np.float64) - 128.0  # centring keeps the expansion below numerically tight
    sq = (x * x).sum(axis=1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (x @ x.T)
    np.maximum(d2, 0.0, out=d2)
    return np.sqrt(d2 / x.shape[1])


def encode_jpeg(img: Image.Image, quality: int = 95) -> bytes:
    """Encode as baseline JPEG without any metadata (EXIF, GPS, ICC thumbnails are all dropped)."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()
