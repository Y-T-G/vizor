"""Replay predictions saved to a pickle.

Useful for two things: testing the refiner without a GPU, and comparing
refiner settings on the same model output instead of paying for inference
again on every run.
"""

import pickle

import numpy as np

from ..boxes import Preds, Tracks
from .base import Model

__all__ = ["Pkl"]

# ultralytics tracked output is [x1, y1, x2, y2, id, conf, cls]
ULTRALYTICS = [0, 1, 2, 3, 5, 6, 4]


class Pkl(Model):
    """One saved list of per-frame boxes, replayed in order.

    Args:
        file: pickle holding a list of (N, 6) or (N, 7) arrays or tensors.
        cols: column order to reindex each frame into ``[x1, y1, x2, y2, conf, cls, id]``.
            Pass ``Pkl.ULTRALYTICS`` for raw ultralytics output. None means the
            arrays are already in Vizor order.
        names: class id to name mapping to hand downstream.

    Frames are handed out in order on each ``track`` or ``find`` call, so the
    file must line up with the video you feed the pipeline.
    """

    ULTRALYTICS = ULTRALYTICS

    def __init__(self, file, cols=None, names=None):
        with open(file, "rb") as f:
            self.frames = list(pickle.load(f))
        self.file = str(file)
        self.cols = cols
        self.names = names
        self.i = 0

    def __len__(self):
        return len(self.frames)

    def reset(self):
        self.i = 0

    def next(self):
        """The next frame as a float32 array, or an empty one past the end."""
        if self.i >= len(self.frames):
            return np.zeros((0, 7), np.float32)
        data = np.asarray(self.frames[self.i], dtype=np.float32)
        self.i += 1
        if data.ndim == 1:
            data = data.reshape(0, 7) if not data.size else data.reshape(1, -1)
        if self.cols is not None and len(data):
            data = data[:, self.cols]
        return data

    def track(self, img=None):
        return Tracks(self.next(), names=self.names)

    def find(self, img=None, names=None):
        return Preds(self.next(), names=names or self.names)

    def __iter__(self):
        self.reset()
        while self.i < len(self.frames):
            yield self.track()

    def __repr__(self):
        return f"Pkl({self.file!r}, {len(self)} frames)"
