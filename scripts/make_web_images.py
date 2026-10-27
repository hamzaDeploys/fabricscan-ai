"""Generate every logo-based image the website needs from one source file.

    python scripts/make_web_images.py

Source: web/public/brand/logo-original.png (transparent PNG). Writes to web/public/:
    logo.png              logo cropped to its content (navbar, footer, report headers)
    favicon.ico           16/32/48 px browser-tab icon
    favicon-32.png        32 px PNG favicon
    apple-touch-icon.png  180 px, on white (iOS does not support transparent icons)
    logo-192.png, logo-512.png   web app manifest icons (keep the original padding)
    og-image.png          1200 x 630 social preview card

Uses Windows' Segoe UI fonts when available and falls back to Pillow's default font elsewhere.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "web" / "public"
SOURCE = PUBLIC / "brand" / "logo-original.png"
DEEP, WHITE = (11, 16, 32), (255, 255, 255)
LOGO_BLUE, LOGO_TEAL = (37, 99, 235), (45, 212, 191)


def font(name: str, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in (Path("C:/Windows/Fonts") / name, Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")):
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default(size)


def cropped(logo: Image.Image, margin: float = 0.04) -> Image.Image:
    """Square crop around the visible pixels, with a small margin, so the logo fills small icons."""
    alpha = np.asarray(logo.getchannel("A"))
    ys, xs = np.nonzero(alpha > 12)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    side = int(max(x1 - x0, y1 - y0) * (1 + 2 * margin))
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    canvas.paste(logo.crop((cx - side // 2, cy - side // 2, cx - side // 2 + side, cy - side // 2 + side)), (0, 0))
    return canvas


def on_background(logo: Image.Image, size: int, background: tuple[int, int, int], padding: float) -> Image.Image:
    inner = int(size * (1 - 2 * padding))
    img = Image.new("RGBA", (size, size), background + (255,))
    mark = logo.resize((inner, inner), Image.Resampling.LANCZOS)
    img.alpha_composite(mark, ((size - inner) // 2, (size - inner) // 2))
    return img.convert("RGB")


def og_image(mark: Image.Image) -> Image.Image:
    img = Image.new("RGBA", (1200, 630), DEEP + (255,))
    grid = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(grid)
    for x in range(0, 1200, 18):
        d.line([(x, 0), (x, 630)], fill=(255, 255, 255, 12))
    for y in range(0, 630, 18):
        d.line([(0, y), (1200, y)], fill=(255, 255, 255, 12))
    glow = Image.new("RGBA", img.size, (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    gd.ellipse([700, -160, 1320, 460], fill=LOGO_BLUE + (140,))
    gd.ellipse([880, 160, 1360, 640], fill=LOGO_TEAL + (90,))
    img = Image.alpha_composite(Image.alpha_composite(img, grid), glow.filter(ImageFilter.GaussianBlur(90)))
    img.alpha_composite(mark.resize((330, 330), Image.Resampling.LANCZOS), (800, 150))
    d = ImageDraw.Draw(img)
    d.text((80, 170), "FabricScan AI", font=font("segoeuib.ttf", 86), fill=WHITE)
    d.text((84, 285), "Textile defect detection with", font=font("segoeui.ttf", 36), fill=(214, 220, 240))
    d.text((84, 330), "automatic defect reports", font=font("segoeui.ttf", 36), fill=(214, 220, 240))
    d.text((84, 400), "Hole · Stain · Broken thread", font=font("segoeui.ttf", 26),
           fill=LOGO_TEAL)
    d.text((84, 530), "CNN from scratch vs MobileNetV2  ·  Grad-CAM  ·  runs in the browser",
           font=font("segoeui.ttf", 24), fill=(150, 160, 190))
    return img.convert("RGB")


def main() -> int:
    if not SOURCE.is_file():
        print(f"Logo not found: {SOURCE}")
        return 1
    original = Image.open(SOURCE).convert("RGBA")
    mark = cropped(original)

    mark.resize((256, 256), Image.Resampling.LANCZOS).save(PUBLIC / "logo.png", optimize=True)
    icon_mark = cropped(original, margin=0.0)
    icon_mark.save(PUBLIC / "favicon.ico", sizes=[(16, 16), (32, 32), (48, 48)])
    icon_mark.resize((32, 32), Image.Resampling.LANCZOS).save(PUBLIC / "favicon-32.png", optimize=True)
    on_background(mark, 180, WHITE, padding=0.08).save(PUBLIC / "apple-touch-icon.png", optimize=True)
    original.resize((192, 192), Image.Resampling.LANCZOS).save(PUBLIC / "logo-192.png", optimize=True)
    original.resize((512, 512), Image.Resampling.LANCZOS).save(PUBLIC / "logo-512.png", optimize=True)
    og_image(mark).save(PUBLIC / "og-image.png", optimize=True)

    old_svg = PUBLIC / "favicon.svg"  # earlier drawn logo, replaced by this one
    if old_svg.exists():
        old_svg.unlink()
    print("Wrote logo.png, favicon.ico, favicon-32.png, apple-touch-icon.png, logo-192.png, logo-512.png, og-image.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
