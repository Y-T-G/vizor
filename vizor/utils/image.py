"""Image helpers. All arrays are BGR, the layout OpenCV gives you."""

import cv2
import numpy as np

__all__ = ["COLORS", "crop", "fit", "label", "montage"]

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


def _cell(cell):
    """A cell size given as one number, or as a (width, height) pair."""
    w, h = (cell, cell) if isinstance(cell, (int, float)) else cell
    return max(1, int(w)), max(1, int(h))


def fit(img, w, h, color=(114, 114, 114)):
    """Resize ``img`` to fill a ``w`` x ``h`` cell without changing its shape.

    The spare space is filled with ``color``, so a tall crop of a person comes
    back with bars at the sides rather than squashed into a square. Returns a
    new array, never a view.
    """
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    src_h, src_w = img.shape[:2]
    if not src_h or not src_w:
        raise ValueError("cannot fit an empty image")
    scale = min(w / src_w, h / src_h)
    new = max(1, round(src_w * scale)), max(1, round(src_h * scale))
    # INTER_AREA is the one that does not alias when shrinking, which is the
    # usual direction here
    small = cv2.resize(img, new, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    out = np.full((h, w, 3), color, np.uint8)
    y, x = (h - new[1]) // 2, (w - new[0]) // 2
    out[y:y + new[1], x:x + new[0]] = small
    return out


def montage(
    imgs: "list[np.ndarray]",
    cols: "int | None" = None,
    cell: "int | tuple[int, int]" = 128,
    pad: int = 4,
    color: "tuple[int, int, int]" = (0, 0, 0),
    fill: "tuple[int, int, int]" = (114, 114, 114),
):
    """Tile images into one BGR grid, reading left to right, top to bottom.

    This is how several crops of one object become a single image a VLM can be
    asked about once. The gutters are drawn in ``color`` and the letterboxing
    inside each cell in ``fill``, two different shades so the model can see
    where one crop ends and the next starts.

    Args:
        imgs: BGR arrays. Empty ones are dropped.
        cols: columns. Defaults to a square-ish grid.
        cell: cell size, one number for a square or a ``(width, height)`` pair.
        pad: gutter in pixels, drawn around the outside as well as between cells.
        color: gutter colour.
        fill: letterbox colour inside a cell.
    """
    imgs = [i for i in imgs if i is not None and i.size]
    if not imgs:
        raise ValueError("montage needs at least one image")
    w, h = _cell(cell)
    cols = max(1, min(int(cols), len(imgs))) if cols else int(np.ceil(np.sqrt(len(imgs))))
    rows = int(np.ceil(len(imgs) / cols))
    pad = max(0, int(pad))
    sheet = np.full((rows * h + (rows + 1) * pad, cols * w + (cols + 1) * pad, 3),
                    color, np.uint8)
    for i, img in enumerate(imgs):
        row, col = divmod(i, cols)
        y, x = pad + row * (h + pad), pad + col * (w + pad)
        sheet[y:y + h, x:x + w] = fit(img, w, h, fill)
    return sheet
