# SPDX-License-Identifier: AGPL-3.0-or-later
#
# This file is NOT part of the vizor package and is not installed by pip.
# The rest of this repository is Apache-2.0. This file is AGPL-3.0-or-later
# because it imports ultralytics, which is AGPL-3.0-or-later, and the two form
# a combined work when you run them. Copy it into your own project only if that
# project can be AGPL-3.0, or buy an Ultralytics Enterprise licence.
#
# https://github.com/ultralytics/ultralytics/blob/main/LICENSE
"""Ultralytics YOLO as a vizor model.

Install ultralytics first, then pass ``YOLO`` to ``Vizor`` as the primary. It
is not exported from ``vizor``, so you import it from here.

    pip install vizor ultralytics

    from yolo import YOLO
    viz = vz.Vizor(YOLO("yolo11n.pt"), vz.Vlm("gpt-4o-mini"), mode="crop")
"""

import numpy as np

from vizor import Model, Preds, Tracks

__all__ = ["YOLO"]

# ultralytics tracked output is [x1, y1, x2, y2, id, conf, cls]
ULTRALYTICS = [0, 1, 2, 3, 5, 6, 4]


class YOLO(Model):
    """A YOLO model. ``track`` makes it a primary, ``find`` a secondary.

    Args:
        model: weights path or an already built ``ultralytics.YOLO``.
        tracker: tracker config used by ``track``.
        kw: forwarded to every ultralytics call, e.g. ``imgsz``, ``device``, ``half``.
    """

    def __init__(self, model="yolo11n.pt", tracker="bytetrack.yaml", **kw):
        import ultralytics

        self.model = ultralytics.YOLO(model) if isinstance(model, str) else model
        self.tracker = tracker
        self.kw = kw
        self.names = getattr(self.model, "names", None)

    @staticmethod
    def _data(result):
        """Result boxes as (N, 7) in Vizor order, filling in ids when absent."""
        data = result.boxes.data
        data = data.cpu().numpy() if hasattr(data, "cpu") else np.asarray(data)
        data = data.astype(np.float32)
        if not len(data):
            return np.zeros((0, 7), np.float32)
        if data.shape[-1] == 7:
            return data[:, ULTRALYTICS]
        return np.column_stack([data[:, :6], np.full(len(data), -1, np.float32)])

    def track(self, img):
        """Detect and track on a BGR frame, keeping tracker state between calls."""
        result = self.model.track(img, persist=True, tracker=self.tracker,
                                  verbose=False, **self.kw)[0]
        self.names = result.names
        return Tracks(self._data(result), names=result.names, img=img)

    def find(self, img, names=None):
        """Detect on a BGR frame without tracking. Used as a full mode secondary."""
        result = self.model.predict(img, verbose=False, **self.kw)[0]
        self.names = result.names
        return Preds(self._data(result), names=names or result.names)

    def name(self, crop, names=None, hint=None):
        """Class of the largest detection in the crop, or None if there is none."""
        preds = self.find(crop, names)
        if not len(preds):
            return None
        areas = (preds.boxes[:, 2] - preds.boxes[:, 0]) * (preds.boxes[:, 3] - preds.boxes[:, 1])
        return int(preds.cls[int(areas.argmax())])

    def reset(self):
        """Drop the tracker state so ids start from 1 again on the next video."""
        predictor = getattr(self.model, "predictor", None)
        for tracker in getattr(predictor, "trackers", None) or []:
            reset = getattr(tracker, "reset", None)
            if callable(reset):
                reset()
