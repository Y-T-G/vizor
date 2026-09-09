import numpy as np
import pytest
from conftest import boxes

from vizor import Vizor
from vizor.boxes import Preds, Tracks
from vizor.models.base import Model
from vizor.utils.image import crop, label


class Fake(Model):
    """Primary that hands out one canned frame of tracks at a time."""

    names = {0: "person", 7: "truck"}

    def __init__(self, frames):
        self.frames = frames
        self.i = 0
        self.reset_calls = 0

    def track(self, img):
        data = self.frames[min(self.i, len(self.frames) - 1)]
        self.i += 1
        return Tracks(data, names=self.names, img=img)

    def reset(self):
        self.reset_calls += 1
        self.i = 0


def test_step_refines(frame):
    primary = Fake([boxes([0, 0, 100, 100, 0.2, 0, 1])])
    secondary = type("S", (Model,), {
        "find": lambda self, img, names=None: Preds(
            np.array([[0, 0, 100, 100, 0.99, 7]], np.float32))})()
    viz = Vizor(primary, secondary, conf=0.5, mode="full")
    out = viz.step(frame)
    assert out.cls[0] == 7
    assert out.name(7) == "truck"
    assert viz.names == primary.names


def test_step_without_a_secondary(frame):
    viz = Vizor(Fake([boxes([0, 0, 10, 10, 0.1, 0, 1])]))
    out = viz.step(frame)
    assert len(out) == 1 and out.cls[0] == 0


def test_reset_reaches_the_primary():
    primary = Fake([boxes([0, 0, 10, 10, 0.1, 0, 1])])
    viz = Vizor(primary)
    viz.reset()
    assert primary.reset_calls == 1


def test_run_over_a_video(tmp_path):
    import cv2

    src = tmp_path / "in.mp4"
    out = tmp_path / "out.mp4"
    writer = cv2.VideoWriter(str(src), cv2.VideoWriter_fourcc(*"mp4v"), 10, (160, 120))
    for _ in range(5):
        writer.write(np.full((120, 160, 3), 40, np.uint8))
    writer.release()

    viz = Vizor(Fake([boxes([10, 10, 60, 60, 0.9, 0, 1])]))
    n = sum(1 for _ in viz.run(str(src), save=str(out)))
    assert n == 5
    assert out.exists() and out.stat().st_size > 0


def test_missing_video():
    with pytest.raises(OSError):
        next(Vizor(Fake([boxes([0, 0, 1, 1, 0.9, 0, 1])])).run("no_such_file.mp4"))


def test_crop_clamps_to_the_frame():
    img = np.zeros((100, 100, 3), np.uint8)
    assert crop(img, [10, 10, 50, 50]).shape[:2] == (48, 48)   # 40px box plus a 4px margin each side
    assert crop(img, [-50, -50, -10, -10]).size == 0
    assert crop(img, [90, 90, 200, 200]).shape[:2] == (21, 21)  # clipped at the edge


def test_label_fallbacks():
    assert label({2: "car"}, 2) == "car"
    assert label(["a", "b"], 1) == "b"
    assert label(None, 4) == "4"
    assert label(["a"], 9) == "9"
