"""End to end run on the cached predictions in nbs/, if they are present.

These are real YOLOv5n tracks refined by real Florence-2 grounding output, so
the test exercises the whole pipeline without needing a GPU or a network.
"""

from pathlib import Path

import numpy as np
import pytest

from vizor import Vizor
from vizor.models.pkl import Pkl

ROOT = Path(__file__).resolve().parents[1]
PRIMARY = ROOT / "nbs" / "yolov5nu_traffic3_preds.pkl"
SECONDARY = ROOT / "nbs" / "Florence-2-base-ft_traffic3_preds.pkl"
VIDEO = ROOT / "nbs" / "traffic3.mp4"

pytestmark = pytest.mark.skipif(
    not (PRIMARY.exists() and SECONDARY.exists()),
    reason="cached predictions not in the checkout",
)

NAMES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


def pipeline(**kw):
    primary = Pkl(PRIMARY, cols=Pkl.ULTRALYTICS, names=NAMES)
    secondary = Pkl(SECONDARY, names=NAMES)
    return Vizor(primary, secondary, names=NAMES, **kw)


def test_replay_changes_some_classes():
    viz = pipeline(conf=0.5, mode="full")
    before = after = 0
    for _ in range(200):
        raw = np.asarray(viz.primary.frames[viz.primary.i], np.float32)
        out = viz.step(None)
        if not len(out):
            continue
        before += int(raw.shape[0])
        after += int((out.cls != raw[:, 6].astype(int)).sum())
    assert before > 0
    assert after > 0, "the secondary never corrected anything"


def test_replay_keeps_the_box_count():
    viz = pipeline(conf=0.5, mode="full")
    for _ in range(50):
        n = len(viz.primary.frames[viz.primary.i])
        assert len(viz.step(None)) == n  # refining never adds or drops boxes


def test_votes_are_bounded():
    viz = pipeline(conf=0.5, mode="full", size=64)
    for _ in range(500):
        viz.step(None)
    assert len(viz.refiner.cache) <= 64


@pytest.mark.skipif(not VIDEO.exists(), reason="traffic3.mp4 not in the checkout")
def test_replay_draws_on_real_frames():
    from vizor.utils.video import Video

    viz = pipeline(conf=0.5, mode="full")
    with Video(str(VIDEO)) as video:
        for i, frame in enumerate(video):
            out = viz.step(frame)
            assert out.draw().shape == frame.shape
            if i >= 20:
                break
