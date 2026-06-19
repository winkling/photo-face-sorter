import os
import cv2
import numpy as np
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

register_heif_opener()

SUPPORTED = {".jpg", ".jpeg", ".png", ".heic", ".heif"}


def load_bgr(path: str) -> np.ndarray:
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def iter_images(directory: str):
    for root, _, files in os.walk(directory):
        for fname in sorted(files):
            ext = os.path.splitext(fname)[1].lower()
            if ext in SUPPORTED:
                yield os.path.join(root, fname)


def iter_other_files(directory: str):
    for root, _, files in os.walk(directory):
        for fname in sorted(files):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in SUPPORTED:
                yield os.path.join(root, fname)


def crop_face(bgr: np.ndarray, bbox, margin: float = 0.10) -> np.ndarray:
    x1, y1, x2, y2 = (int(v) for v in bbox)
    h, w = bgr.shape[:2]
    dx = int((x2 - x1) * margin)
    dy = int((y2 - y1) * margin)
    x1 = max(0, x1 - dx)
    y1 = max(0, y1 - dy)
    x2 = min(w, x2 + dx)
    y2 = min(h, y2 + dy)
    return bgr[y1:y2, x1:x2]
