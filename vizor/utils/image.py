"""Image helpers. All arrays are BGR, the layout OpenCV gives you."""

import numpy as np

__all__ = ["COLORS", "crop", "label"]

# BGR, picked to stay apart from each other when drawn side by side
COLORS = [
    (56, 56, 255),
    (56, 255, 56),
    (255, 157, 56),
    (56, 200, 255),
    (255, 56, 200),
    (0, 215, 255),
    (200, 56, 255),
    (128, 255, 128),
    (255, 128, 0),
    (100, 100, 255),
]


def label(names, cls):
    """Class id to string. Works with a list, a dict, or nothing at all."""
    cls = int(cls)
    try:
        if isinstance(names, dict):
            return str(names[cls])
        if names is not None:
            return str(names[cls])
    except (KeyError, IndexError):
        pass
    return str(cls)


def crop(img, box, border=0.1):
    """Cut ``box`` out of ``img`` with a margin of ``border`` x the shorter side.

    Returns a BGR view. It can be empty if the box falls outside the frame,
    so check ``.size`` before feeding it to a model.
    """
    x1, y1, x2, y2 = (int(round(float(v))) for v in box[:4])
    margin = int(border * min(x2 - x1, y2 - y1))
    h, w = img.shape[:2]
    x1, y1 = max(0, x1 - margin), max(0, y1 - margin)
    x2, y2 = min(w, x2 + margin), min(h, y2 + margin)
    if x2 <= x1 or y2 <= y1:
        return np.zeros((0, 0, img.shape[2] if img.ndim == 3 else 1), img.dtype)
    return img[y1:y2, x1:x2]
