"""Drawing helpers shared by the label tool and the prediction visualiser."""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Colours (RGB)
GREEN = (30, 160, 80)
RED = (210, 50, 40)
AMBER = (235, 160, 20)
WHITE = (255, 255, 255)
DARK = (28, 30, 34)

# Fonts that can render Hangul come first; the rest are Latin-only fallbacks.
_KOREAN_FONTS = [
    "C:/Windows/Fonts/malgun.ttf",
    "C:/Windows/Fonts/malgunbd.ttf",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
]
_LATIN_FONTS = [
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/segoeui.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
]


@lru_cache(maxsize=None)
def _font_path() -> tuple[str | None, bool]:
    """Return (path, supports_korean)."""
    for p in _KOREAN_FONTS:
        if Path(p).exists():
            return p, True
    for p in _LATIN_FONTS:
        if Path(p).exists():
            return p, False
    return None, False


def korean_font_available() -> bool:
    return _font_path()[1]


@lru_cache(maxsize=None)
def get_font(size: int) -> ImageFont.ImageFont:
    path, _ = _font_path()
    if path is not None:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    try:  # Pillow >= 10.1 can scale the built-in bitmap font
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def text_size(draw: ImageDraw.ImageDraw, text: str, font) -> tuple[int, int]:
    l, t, r, b = draw.textbbox((0, 0), text, font=font)
    return r - l, b - t


def pil_to_bgr(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB"))[:, :, ::-1].copy()


def fit_within(img: Image.Image, max_w: int, max_h: int) -> Image.Image:
    w, h = img.size
    s = min(max_w / w, max_h / h, 1.0)
    if s < 1.0:
        img = img.resize((max(1, int(w * s)), max(1, int(h * s))), Image.BILINEAR)
    return img


def annotate_prediction(img: Image.Image, label: str, confidence: float) -> Image.Image:
    """Draw a coloured frame (green = normal, red = foul) + banner with label and confidence."""
    img = img.convert("RGB").copy()
    w, h = img.size
    colour = GREEN if label == "normal" else RED

    border = max(4, int(min(w, h) * 0.015))
    draw = ImageDraw.Draw(img)
    for i in range(border):
        draw.rectangle([i, i, w - 1 - i, h - 1 - i], outline=colour)

    font = get_font(max(16, int(min(w, h) * 0.06)))
    text = f"{label.upper()}  {confidence * 100:.0f}%"
    tw, th = text_size(draw, text, font)
    pad = max(6, th // 3)
    banner_h = th + 2 * pad
    draw.rectangle([0, 0, w, banner_h], fill=colour)
    draw.text((pad, pad - th * 0.1), text, fill=WHITE, font=font)
    return img


def is_windows() -> bool:
    return sys.platform.startswith("win")
