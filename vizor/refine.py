"""The refiner: a fast tracker corrected by a slow, accurate model."""

from typing import TYPE_CHECKING

import numpy as np

from .boxes import Preds, Tracks, iou
from .utils.image import crop, label
from .vote import Vote

if TYPE_CHECKING:
    from .models.base import Model

__all__ = ["Refiner"]

# the old notebook names, kept working
MODES = {"full": "full", "image": "full", "crop": "crop", "instance": "crop"}


class Refiner:
    """Correct low confidence tracks with a second model and cache the answers.

    Two modes:

    ``full``
        Run the secondary on the whole frame, match its boxes to the tracks by
        IoU, and take its box, confidence and class. Best when the secondary
        is a grounding model or a heavier detector.
    ``crop``
        Cut each low confidence track out of the frame and ask the secondary
        what it is. Best when the secondary is a chat VLM that classifies but
        does not localise.

    Every answer is a vote against the track id, so a track keeps its corrected
    class on later frames without the secondary running again.

    Args:
        model: the secondary. Needs ``find`` in full mode, ``name`` in crop mode.
        conf: tracks at or below this confidence are sent to the secondary.
        mode: ``"full"`` or ``"crop"``.
        iou: minimum IoU to match a secondary box to a track, full mode only.
        names: class id to name mapping. Falls back to whatever the primary reports.
        votes: stop asking about a track once it has this many votes, crop mode only.
        best: keep only the best IoU match per track instead of every match above ``iou``.
        size: how many track ids to keep in the vote cache.
        hist: how many votes to keep per track id.
    """

    def __init__(
        self,
        model: "Model | None" = None,
        conf: float = 0.5,
        mode: str = "full",
        iou: float = 0.5,
        names: "dict[int, str] | list[str] | None" = None,
        votes: int = 1,
        best: bool = True,
        size: int = 256,
        hist: int = 25,
    ):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {sorted(set(MODES))}, got {mode!r}")
        self.model = model
        self.conf = float(conf)
        self.mode = MODES[mode]
        self.iou = float(iou)
        self.names = names
        self.votes = int(votes)
        self.best = bool(best)
        self.cache = Vote(size=size, hist=hist)

    def reset(self):
        """Forget every vote. Call this between videos."""
        self.cache.clear()

    def run(self, tracks, img=None, preds=None):
        """Refine ``tracks`` in place and return them.

        ``img`` is the current frame. ``preds`` lets you supply the secondary's
        output yourself, which skips the model call entirely.
        """
        if not isinstance(tracks, Tracks):
            tracks = Tracks(tracks)
        if self.names is None:
            self.names = tracks.names
        if len(tracks):
            if self.mode == "full":
                self._full(tracks, img, preds)
            else:
                self._crop(tracks, img)
            self._apply(tracks)
        tracks.names = self.names if self.names is not None else tracks.names
        tracks.img = img if img is not None else tracks.img
        return tracks

    __call__ = run

    # modes ----------------------------------------------------------------
    def _full(self, tracks, img, preds):
        # nothing is in doubt, so the secondary has nothing to add
        if not (tracks.conf <= self.conf).any():
            return
        if preds is None:
            if self.model is None:
                return
            preds = self.model.find(img, self.names)
        if not isinstance(preds, Tracks):
            preds = Preds(preds)
        if not len(preds):
            return
        m = iou(tracks.boxes, preds.boxes)
        for i, row in enumerate(m):
            idx = np.flatnonzero(row >= self.iou)
            if not len(idx):
                continue
            if self.best:
                idx = idx[[row[idx].argmax()]]
            track = tracks[i]
            for j in idx:
                pred = preds[int(j)]
                track.box, track.conf = pred.box, pred.conf
                self.cache.add(track.id, pred.cls)
            tracks[i] = track

    def _crop(self, tracks, img):
        if self.model is None or img is None:
            return
        for track in tracks:
            if track.conf > self.conf or self.cache.count(track.id) >= self.votes:
                continue
            patch = crop(img, track.box)
            if patch.size == 0:
                continue
            cls = self.model.name(patch, self.names, hint=label(self.names, track.cls))
            if cls is not None:
                self.cache.add(track.id, cls)

    def _apply(self, tracks):
        """Overwrite each class with the running majority vote for its track id."""
        for i, track in enumerate(tracks):
            cls = self.cache.get(track.id)
            if cls is not None and cls != track.cls:
                track.cls = cls
                tracks[i] = track
