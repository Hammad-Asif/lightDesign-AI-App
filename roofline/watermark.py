"""Stock-photo watermark check. Watermarks pollute line detection, so the harness warns loudly.

Three signals, any one of which raises the warning:
  1. the photo's name is listed in eval/watermarked.txt (the reliable one: you know your photos),
  2. the file name mentions a stock agency or "watermark",
  3. a tiled overlay is detected: thin bright strokes whose autocorrelation has an isolated,
     prominent peak at a large offset (a glyph repeated on a lattice).

Signal 3 catches bold tiled watermarks. Faint ones (Dreamstime's semi-transparent spirals, for
example) are NOT reliably detectable this way: their lattice peak is no stronger than structure
in clean photos. List such photos in eval/watermarked.txt.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

NAME_HINTS = ("watermark", "dreamstime", "shutterstock", "istock", "alamy", "gettyimages", "depositphotos", "123rf", "adobestock")
PROMINENCE_THRESHOLD = 0.12


def lattice_prominence(image_bgr: np.ndarray, size: int = 768) -> float:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    s = size / max(gray.shape)
    gray = cv2.resize(gray, (round(gray.shape[1] * s), round(gray.shape[0] * s)), interpolation=cv2.INTER_AREA).astype(np.float32)
    strokes = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    strokes = np.clip(strokes, 0, 8) / 8.0          # a faint overlay counts as much as a bright one
    strokes -= strokes.mean()
    h, w = strokes.shape
    spectrum = np.fft.fft2(strokes)
    corr = np.fft.fftshift(np.real(np.fft.ifft2(spectrum * np.conj(spectrum))))
    corr /= corr.max() + 1e-9
    yy, xx = np.mgrid[:h, :w]
    dy, dx = yy - h // 2, xx - w // 2
    corr = corr / np.maximum((1 - np.abs(dx) / w) * (1 - np.abs(dy) / h), 0.2)   # less overlap at large offsets
    far = (np.hypot(dx, dy) > 0.15 * max(h, w)) & (np.abs(dx) <= 0.45 * w) & (np.abs(dy) <= 0.45 * h)
    prominence = cv2.GaussianBlur(corr.astype(np.float32), (0, 0), 1.5) - cv2.GaussianBlur(corr.astype(np.float32), (0, 0), 9)
    return float(prominence[far].max())


def is_watermarked(image_bgr: np.ndarray, name: str = "", listed: Optional[set[str]] = None) -> tuple[bool, str]:
    stem = Path(name).stem.lower()
    if listed and stem in listed:
        return True, "listed in eval/watermarked.txt"
    if any(hint in stem for hint in NAME_HINTS):
        return True, "file name mentions a stock agency"
    p = lattice_prominence(image_bgr)
    if p >= PROMINENCE_THRESHOLD:
        return True, f"tiled overlay detected (lattice prominence {p:.2f})"
    return False, f"no tiled overlay detected (lattice prominence {p:.2f}; faint watermarks are not detectable)"
