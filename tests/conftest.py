import numpy as np
import pytest

from vizor.models.base import Model

NAMES = {0: "person", 1: "bicycle", 2: "car", 7: "truck"}


def boxes(*rows):
    """Build an (N, 7) array from [x1, y1, x2, y2, conf, cls, id] rows."""
    return np.array(rows, np.float32).reshape(-1, 7)


@pytest.fixture
def names():
    return dict(NAMES)


@pytest.fixture
def frame():
    return np.full((480, 640, 3), 32, np.uint8)


class Fixed(Model):
    """Secondary that always answers the same thing, and counts the calls."""

    def __init__(self, cls=7, preds=None):
        self.cls = cls
        self.preds = preds
        self.calls = 0

    def find(self, img, names=None):
        from vizor.boxes import Preds

        self.calls += 1
        return Preds(self.preds, names=names)

    def name(self, crop, names=None, hint=None):
        self.calls += 1
        return self.cls


@pytest.fixture
def fixed():
    return Fixed
