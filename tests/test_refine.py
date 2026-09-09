import numpy as np
import pytest
from conftest import boxes

from vizor.boxes import Preds, Tracks
from vizor.refine import Refiner


def test_bad_mode():
    with pytest.raises(ValueError):
        Refiner(mode="sideways")


def test_mode_aliases():
    assert Refiner(mode="image").mode == "full"
    assert Refiner(mode="instance").mode == "crop"


def test_full_fixes_class_and_box(names):
    tracks = Tracks(boxes([0, 0, 100, 100, 0.3, 0, 1]), names=names)
    preds = Preds(np.array([[2, 2, 102, 102, 0.95, 7]], np.float32))
    out = Refiner(conf=0.5, mode="full").run(tracks, preds=preds)
    assert out.cls[0] == 7
    assert out.conf[0] == pytest.approx(0.95)
    assert out.boxes[0].tolist() == [2, 2, 102, 102]


def test_full_skips_confident_tracks():
    tracks = Tracks(boxes([0, 0, 100, 100, 0.9, 0, 1]))
    preds = Preds(np.array([[0, 0, 100, 100, 0.95, 7]], np.float32))
    out = Refiner(conf=0.5, mode="full").run(tracks, preds=preds)
    assert out.cls[0] == 0  # left alone, so the secondary costs nothing


def test_full_needs_enough_overlap():
    tracks = Tracks(boxes([0, 0, 100, 100, 0.3, 0, 1]))
    preds = Preds(np.array([[200, 200, 300, 300, 0.95, 7]], np.float32))
    out = Refiner(conf=0.5, mode="full", iou=0.5).run(tracks, preds=preds)
    assert out.cls[0] == 0


def test_full_calls_the_model_when_no_preds_given(fixed):
    model = fixed(preds=np.array([[0, 0, 100, 100, 0.9, 7]], np.float32))
    ref = Refiner(model, conf=0.5, mode="full")
    ref.run(Tracks(boxes([0, 0, 100, 100, 0.3, 0, 1])), img=np.zeros((10, 10, 3), np.uint8))
    assert model.calls == 1
    # a confident frame must not reach the model
    ref.run(Tracks(boxes([0, 0, 100, 100, 0.99, 0, 2])), img=np.zeros((10, 10, 3), np.uint8))
    assert model.calls == 1


def test_the_vote_survives_later_frames():
    ref = Refiner(conf=0.5, mode="full")
    preds = Preds(np.array([[0, 0, 100, 100, 0.95, 7]], np.float32))
    ref.run(Tracks(boxes([0, 0, 100, 100, 0.3, 0, 1])), preds=preds)
    # next frame: no secondary output at all, but track 1 keeps the fixed class
    out = ref.run(Tracks(boxes([1, 1, 101, 101, 0.3, 0, 1])))
    assert out.cls[0] == 7


def test_reset_clears_the_cache():
    ref = Refiner(conf=0.5, mode="full")
    preds = Preds(np.array([[0, 0, 100, 100, 0.95, 7]], np.float32))
    ref.run(Tracks(boxes([0, 0, 100, 100, 0.3, 0, 1])), preds=preds)
    ref.reset()
    assert ref.run(Tracks(boxes([0, 0, 100, 100, 0.3, 0, 1]))).cls[0] == 0


def test_untracked_boxes_do_not_share_votes():
    ref = Refiner(conf=0.5, mode="full")
    preds = Preds(np.array([[0, 0, 100, 100, 0.95, 7]], np.float32))
    ref.run(Tracks(boxes([0, 0, 100, 100, 0.3, 0, -1])), preds=preds)
    out = ref.run(Tracks(boxes([500, 500, 600, 600, 0.3, 1, -1])))
    assert out.cls[0] == 1


def test_best_match_only():
    tracks = Tracks(boxes([0, 0, 100, 100, 0.3, 0, 1]))
    preds = Preds(np.array([[0, 0, 100, 100, 0.9, 7],
                            [0, 0, 90, 90, 0.9, 2]], np.float32))
    out = Refiner(conf=0.5, mode="full", best=True).run(tracks, preds=preds)
    assert out.cls[0] == 7  # the higher IoU box wins


def test_crop_asks_once_per_track(frame, fixed, names):
    model = fixed(cls=7)
    ref = Refiner(model, conf=0.5, mode="crop", names=names, votes=1)
    for _ in range(5):
        out = ref.run(Tracks(boxes([10, 10, 110, 110, 0.3, 0, 1])), img=frame)
    assert model.calls == 1
    assert out.cls[0] == 7


def test_crop_votes_more_than_once(frame, fixed, names):
    model = fixed(cls=7)
    ref = Refiner(model, conf=0.5, mode="crop", names=names, votes=3)
    for _ in range(5):
        ref.run(Tracks(boxes([10, 10, 110, 110, 0.3, 0, 1])), img=frame)
    assert model.calls == 3


def test_crop_skips_boxes_outside_the_frame(frame, fixed):
    model = fixed(cls=7)
    Refiner(model, conf=0.5, mode="crop").run(
        Tracks(boxes([900, 900, 1000, 1000, 0.3, 0, 1])), img=frame)
    assert model.calls == 0


def test_empty_tracks_are_fine(frame, fixed):
    model = fixed(cls=7)
    out = Refiner(model, conf=0.5, mode="crop").run(Tracks(), img=frame)
    assert len(out) == 0 and model.calls == 0


def test_raw_array_input():
    out = Refiner(conf=0.5).run(np.array([[0, 0, 10, 10, 0.9, 1, 2]], np.float32))
    assert isinstance(out, Tracks) and out.cls[0] == 1
