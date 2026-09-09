"""Box containers shared by every part of Vizor.

One layout is used everywhere: a float32 array of shape (N, 7) holding
``[x1, y1, x2, y2, conf, cls, id]``. Untracked rows carry ``id = -1``.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from .utils.image import COLORS, label

__all__ = ["Track", "Tracks", "Preds", "iou"]

COLS = 7


def _as_data(data):
    """Coerce anything array-like to a float32 (N, 7) array."""
    if data is None:
        return np.zeros((0, COLS), np.float32)
    data = np.asarray(data, dtype=np.float32)
    if data.ndim == 1:
        data = data.reshape(1, -1) if data.size else data.reshape(0, COLS)
    if data.shape[-1] == COLS - 1:  # no track id column
        data = np.column_stack([data, np.full(len(data), -1, np.float32)])
    if data.shape[-1] != COLS:
        raise ValueError(f"expected {COLS - 1} or {COLS} columns, got {data.shape[-1]}")
    return np.ascontiguousarray(data)


def iou(a, b):
    """Pairwise IoU between two sets of xyxy boxes, shape (len(a), len(b))."""
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)), np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:4], b[None, :, 2:4])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = ((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]))[:, None]
    area_b = ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))[None, :]
    return inter / np.clip(area_a + area_b - inter, 1e-9, None)


@dataclass
class Track:
    """One box. ``box`` is xyxy, ``id`` is -1 when the box is untracked."""

    box: np.ndarray
    conf: float
    cls: int
    id: int

    @property
    def wh(self):
        """Width and height of the box."""
        return self.box[2] - self.box[0], self.box[3] - self.box[1]

    @property
    def area(self):
        """Box area in pixels, clamped at zero for an inverted box."""
        w, h = self.wh
        return max(0.0, float(w)) * max(0.0, float(h))


class Tracks:
    """A sliceable view over an (N, 7) box array.

    ``names`` maps class ids to strings (list or dict, either works) and
    ``img`` is the frame the boxes came from, so ``draw()`` needs no argument.
    """

    def __init__(self, data=None, names=None, img=None):
        self.data = _as_data(data)
        self.names = names
        self.img = img

    # column views ---------------------------------------------------------
    @property
    def boxes(self):
        """View of the ``(N, 4)`` xyxy columns. Writes go through to ``data``."""
        return self.data[:, :4]

    @property
    def conf(self):
        """View of the ``(N,)`` confidence column."""
        return self.data[:, 4]

    @property
    def cls(self):
        """Class ids as ``(N,)`` int. This is a copy, so writes do not go through."""
        return self.data[:, 5].astype(int)

    @property
    def ids(self):
        """Track ids as ``(N,)`` int, ``-1`` where untracked. A copy, like ``cls``."""
        return self.data[:, 6].astype(int)

    # sequence protocol ----------------------------------------------------
    def __len__(self):
        return len(self.data)

    def __bool__(self):
        return len(self.data) > 0

    def __getitem__(self, idx):
        if isinstance(idx, (int, np.integer)):
            row = self.data[idx]
            return Track(row[:4], float(row[4]), int(row[5]), int(row[6]))
        # slices and masks copy, so the result is a detached Tracks
        return Tracks(self.data[idx], self.names, self.img)

    def __setitem__(self, idx, track):
        if isinstance(track, Track):
            track = [*track.box, track.conf, track.cls, track.id]
        self.data[idx] = track

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __repr__(self):
        return f"{type(self).__name__}({len(self)} boxes)"

    # helpers --------------------------------------------------------------
    def name(self, cls):
        """Class id to string, falling back to the id itself."""
        return label(self.names, cls)

    def copy(self):
        """A deep copy of the rows, sharing the names mapping and the source frame."""
        return type(self)(self.data.copy(), self.names, self.img)

    def draw(self, img=None, scale=None, thick=None):
        """Draw the boxes on ``img`` (defaults to the source frame) and return it."""
        img = self.img if img is None else img
        if img is None:
            raise ValueError("no image to draw on")
        scale = max(0.4, min(1.0, img.shape[0] / 900)) if scale is None else scale
        thick = max(1, round(scale * 2)) if thick is None else thick
        font = cv2.FONT_HERSHEY_SIMPLEX
        for t in self:
            x1, y1, x2, y2 = (int(round(v)) for v in t.box)
            color = COLORS[t.cls % len(COLORS)]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)
            text = self.name(t.cls)
            if t.id >= 0:
                text = f"{t.id}:{text}"
            text = f"{text} {t.conf:.2f}"
            (tw, th), base = cv2.getTextSize(text, font, scale, thick)
            top = max(0, y1 - th - base)
            cv2.rectangle(img, (x1, top), (x1 + tw, top + th + base), color, -1)
            cv2.putText(img, text, (x1, top + th), font, scale, (255, 255, 255),
                        thick, lineType=cv2.LINE_AA)
        return img


class Preds(Tracks):
    """Detections without track ids. Six-column input is padded with ``id = -1``."""
