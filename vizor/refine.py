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
        workers: run the secondary on this many background threads instead of
            blocking the frame loop. 0, the default, blocks. Crop mode only.
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
        workers: int = 0,
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
        self.workers = max(0, int(workers))
        self.pool = None
        self.jobs = []    # (track ids, future) still in flight
        self.busy = set()  # track ids the secondary is already looking at

    def _open(self):
        """The thread pool, built on first use so a blocking refiner starts none."""
        if self.pool is None and self.workers:
            from concurrent.futures import ThreadPoolExecutor

            self.pool = ThreadPoolExecutor(self.workers, thread_name_prefix="vizor")
        return self.pool

    def wait(self):
        """Block until every request in flight has come back, then bank the votes.

        Only useful with ``workers``. The votes land too late for the frames that
        triggered them, but they are there for whatever you refine next.
        """
        for _, fut in list(self.jobs):
            fut.exception()
        self._drain()

    def close(self):
        """Stop the workers. Anything still in flight is dropped."""
        self.jobs.clear()
        self.busy.clear()
        if self.pool is not None:
            self.pool.shutdown(wait=False, cancel_futures=True)
            self.pool = None

    def reset(self):
        """Forget every vote and drop anything in flight. Call this between videos."""
        self.cache.clear()
        for _, fut in self.jobs:
            fut.cancel()
        self.jobs.clear()
        self.busy.clear()

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
        # answers that came back since the last frame, applied before we ask again
        self._drain()
        if self.model is None or img is None:
            return
        # collect the whole frame's doubtful crops first, so a model that can
        # answer about several at once gets the chance to
        ids, rows, crops, hints = [], [], [], []
        for i, track in enumerate(tracks):
            if track.conf > self.conf or not self._due(track.id):
                continue
            patch = crop(img, track.box)
            if patch.size == 0:
                continue
            ids.append(track.id)
            rows.append(i)
            crops.append(patch)
            hints.append(label(self.names, track.cls))
        if not crops:
            return
        # Model.batch falls back to one name() call per crop, so a model that
        # only implements name still works and behaves exactly as before
        pool = self._open()
        if pool is None:
            self._store(self.model.batch(crops, self.names, hints), ids, rows, tracks)
            return
        self.busy.update(ids)
        self.jobs.append((ids, pool.submit(self.model.batch, crops, self.names, hints)))

    def _due(self, id):
        """Is this track worth asking about again?"""
        if id < 0:
            # untracked boxes share id -1, so a vote for one would be a vote for
            # all of them. They are answered inline instead, which a background
            # worker cannot do because the frame is gone by the time it replies.
            return self.pool is None and self.workers == 0
        return self.cache.count(id) < self.votes and id not in self.busy

    def _drain(self):
        """Bank the votes from any request that has finished."""
        if not self.jobs:
            return
        left = []
        for ids, fut in self.jobs:
            if not fut.done():
                left.append((ids, fut))
                continue
            self.busy.difference_update(ids)
            self._store(fut.result(), ids)  # a worker's error surfaces here
        self.jobs = left

    def _store(self, out, ids, rows=None, tracks=None):
        """One answer per crop: a vote for a track, or a direct write if untracked."""
        for i, (id, cls) in enumerate(zip(ids, out)):
            if cls is None:
                continue
            if id >= 0:
                self.cache.add(id, cls)
            elif tracks is not None:
                # nothing to remember it by, so write it now or lose it
                track = tracks[rows[i]]
                track.cls = cls
                tracks[rows[i]] = track

    def _apply(self, tracks):
        """Overwrite each class with the running majority vote for its track id."""
        for i, track in enumerate(tracks):
            cls = self.cache.get(track.id)
            if cls is not None and cls != track.cls:
                track.cls = cls
                tracks[i] = track
