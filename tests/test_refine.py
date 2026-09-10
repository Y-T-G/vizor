import numpy as np
import pytest
from conftest import boxes

from vizor.boxes import Preds, Tracks
from vizor.models.base import Model
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


def test_crop_asks_the_secondary_once_per_frame_not_once_per_box(frame, names):
    """A model that can batch sees the whole frame's crops in one call."""

    class Batching(Model):
        def __init__(self):
            self.calls = 0
            self.sizes = []

        def batch(self, crops, names=None, hints=None):
            self.calls += 1
            self.sizes.append(len(crops))
            return [7] * len(crops)

    model = Batching()
    r = Refiner(model, conf=0.5, mode="crop", names=names)
    tracks = Tracks(boxes(
        [0, 0, 40, 40, 0.1, 0, 1],
        [50, 50, 90, 90, 0.2, 0, 2],
        [100, 100, 140, 140, 0.3, 0, 3],
    ), names=names)
    out = r.run(tracks, img=frame)

    assert model.calls == 1 and model.sizes == [3]
    assert out.cls.tolist() == [7, 7, 7]


def test_crop_passes_the_primary_guess_as_a_hint_for_each_crop(frame, names):
    seen = {}

    class Hints(Model):
        def batch(self, crops, names=None, hints=None):
            seen["hints"] = list(hints)
            return [None] * len(crops)

    r = Refiner(Hints(), conf=0.5, mode="crop", names=names)
    r.run(Tracks(boxes(
        [0, 0, 40, 40, 0.1, 2, 1],
        [50, 50, 90, 90, 0.2, 7, 2],
    ), names=names), img=frame)
    assert seen["hints"] == ["car", "truck"]


def test_crop_skips_the_secondary_when_nothing_is_doubtful(frame, names):
    class Never(Model):
        def batch(self, crops, names=None, hints=None):
            raise AssertionError("should not be called")

    r = Refiner(Never(), conf=0.5, mode="crop", names=names)
    r.run(Tracks(boxes([0, 0, 40, 40, 0.9, 2, 1]), names=names), img=frame)


def test_crop_still_works_for_a_model_that_only_has_name(frame, names, fixed):
    """The default Model.batch loops, so nothing that worked before breaks."""
    model = fixed(cls=7)
    r = Refiner(model, conf=0.5, mode="crop", names=names)
    out = r.run(Tracks(boxes(
        [0, 0, 40, 40, 0.1, 0, 1],
        [50, 50, 90, 90, 0.2, 0, 2],
    ), names=names), img=frame)
    assert model.calls == 2  # one name() per crop
    assert out.cls.tolist() == [7, 7]


# untracked boxes -------------------------------------------------------


def test_untracked_boxes_are_refined_on_their_own_frame(frame, names, fixed):
    """id = -1 has nothing to cache under, so the answer is written straight in."""
    model = fixed(cls=7)
    r = Refiner(model, conf=0.5, mode="crop", names=names)
    for _ in range(3):
        out = r.run(Tracks(boxes([0, 0, 40, 40, 0.2, 0, -1]), names=names), img=frame)
        assert out.cls.tolist() == [7]
    # nothing to remember them by, so they cost a call every frame
    assert model.calls == 3
    assert len(r.cache) == 0


def test_untracked_boxes_do_not_share_one_vote(frame, names):
    """Two id = -1 boxes get their own answers, not one vote pooled across both."""

    class PerCrop(Model):
        def batch(self, crops, names=None, hints=None):
            return [2, 7]

    r = Refiner(PerCrop(), conf=0.5, mode="crop", names=names)
    out = r.run(Tracks(boxes(
        [0, 0, 40, 40, 0.2, 0, -1],
        [50, 50, 90, 90, 0.2, 0, -1],
    ), names=names), img=frame)
    assert out.cls.tolist() == [2, 7]


# background workers ----------------------------------------------------


class Gated(Model):
    """A secondary that answers only when the test lets it."""

    def __init__(self, cls=7):
        import threading

        self.cls = cls
        self.calls = 0
        self.gate = threading.Event()
        self.entered = threading.Event()

    def batch(self, crops, names=None, hints=None):
        self.calls += 1
        self.entered.set()
        self.gate.wait(5)
        return [self.cls] * len(crops)


def doubtful(names, id=1):
    return Tracks(boxes([0, 0, 40, 40, 0.2, 0, id]), names=names)


def test_workers_do_not_block_the_frame_loop(frame, names):
    model = Gated(cls=7)
    r = Refiner(model, conf=0.5, mode="crop", names=names, workers=1)
    try:
        assert model.entered.wait(0) is False
        out = r.run(doubtful(names), img=frame)
        assert out.cls.tolist() == [0]      # the request has not answered yet
        assert model.entered.wait(5)        # but it did go out
        model.gate.set()
        r.wait()
        out = r.run(doubtful(names), img=frame)
        assert out.cls.tolist() == [7]      # a later frame picks up the vote
    finally:
        model.gate.set()
        r.close()


def test_workers_ask_once_while_a_request_is_in_flight(frame, names):
    model = Gated()
    r = Refiner(model, conf=0.5, mode="crop", names=names, workers=1)
    try:
        for _ in range(10):
            r.run(doubtful(names), img=frame)
        assert model.entered.wait(5)
        assert model.calls == 1             # not ten
    finally:
        model.gate.set()
        r.close()


def test_workers_skip_untracked_boxes(frame, names):
    """A late answer cannot reach an id = -1 box, so it is never asked about."""
    model = Gated()
    r = Refiner(model, conf=0.5, mode="crop", names=names, workers=1)
    try:
        out = r.run(Tracks(boxes([0, 0, 40, 40, 0.2, 0, -1]), names=names), img=frame)
        assert out.cls.tolist() == [0]
        assert model.calls == 0
    finally:
        model.gate.set()
        r.close()


def test_a_worker_error_surfaces_on_the_main_thread(frame, names):
    class Broken(Model):
        def batch(self, crops, names=None, hints=None):
            raise RuntimeError("the api said no")

    r = Refiner(Broken(), conf=0.5, mode="crop", names=names, workers=1)
    try:
        r.run(doubtful(names), img=frame)
        with pytest.raises(RuntimeError, match="the api said no"):
            r.wait()
    finally:
        r.close()


def test_reset_and_close_drop_work_in_flight(frame, names):
    model = Gated()
    r = Refiner(model, conf=0.5, mode="crop", names=names, workers=1)
    r.run(doubtful(names), img=frame)
    assert r.jobs and r.busy
    r.reset()
    assert not r.jobs and not r.busy and len(r.cache) == 0
    model.gate.set()
    r.close()
    assert r.pool is None


def test_workers_zero_never_starts_a_pool(frame, names, fixed):
    r = Refiner(fixed(cls=7), conf=0.5, mode="crop", names=names)
    r.run(doubtful(names), img=frame)
    assert r.pool is None and not r.jobs


# collage mode ------------------------------------------------------------
class Sheets(Model):
    """A collage secondary that records what it was handed."""

    def __init__(self, cls=1):
        self.cls = cls
        self.calls = []

    def grid(self, collages, names=None, hints=None, tiles=1):
        self.calls.append({"n": len(collages), "shape": collages[0].shape,
                           "tiles": tiles, "hints": list(hints)})
        return [self.cls] * len(collages)


def person(id=1, conf=0.9, cls=0, names=None):
    return Tracks(boxes([10, 10, 50, 90, conf, cls, id]), names=names)


MENU = {0: "man", 1: "woman"}


def test_collage_waits_for_enough_samples(frame):
    model = Sheets()
    r = Refiner(model, conf=1.0, mode="collage", names=MENU, samples=3, every=4)
    seen = [int(r.run(person(), img=frame).cls[0]) for _ in range(12)]
    # sampled on frames 0, 4 and 8, so the answer lands on 8 and holds after it
    assert seen == [0] * 8 + [1] * 4
    assert len(model.calls) == 1
    assert model.calls[0]["tiles"] == 3


def test_collage_asks_once_per_track(frame):
    model = Sheets()
    r = Refiner(model, conf=1.0, mode="collage", names=MENU, samples=2, every=0)
    for _ in range(30):
        r.run(person(), img=frame)
    assert len(model.calls) == 1  # votes=1, so one collage covers the track for life


def test_collage_keeps_asking_when_votes_is_raised(frame):
    model = Sheets()
    r = Refiner(model, conf=1.0, mode="collage", names=MENU, samples=2, every=0, votes=3)
    for _ in range(30):
        r.run(person(), img=frame)
    assert len(model.calls) == 3


def test_collage_spaces_its_samples(frame):
    model = Sheets()
    r = Refiner(model, conf=1.0, mode="collage", names=MENU, samples=2, every=10)
    for _ in range(10):
        r.run(person(), img=frame)
    assert model.calls == []  # only frames 0 and 10 count, and 10 has not come
    r.run(person(), img=frame)
    assert len(model.calls) == 1


def test_collage_gathers_each_track_separately(frame):
    model = Sheets()
    r = Refiner(model, conf=1.0, mode="collage", names=MENU, samples=2, every=0)
    two = Tracks(boxes(
        [10, 10, 50, 90, 0.9, 0, 1],
        [60, 10, 100, 90, 0.9, 0, 2],
    ))
    r.run(two, img=frame)
    out = r.run(two, img=frame)
    assert model.calls[0]["n"] == 2       # both matured on the same frame
    assert out.cls.tolist() == [1, 1]


def test_collage_skips_untracked_boxes(frame):
    model = Sheets()
    r = Refiner(model, conf=1.0, mode="collage", names=MENU, samples=2, every=0)
    for _ in range(6):
        out = r.run(person(id=-1), img=frame)
    assert model.calls == []
    assert out.cls.tolist() == [0]


def test_collage_skips_confident_tracks(frame):
    model = Sheets()
    r = Refiner(model, conf=0.5, mode="collage", names=MENU, samples=2, every=0)
    for _ in range(6):
        r.run(person(conf=0.9), img=frame)
    assert model.calls == []


def test_collage_hint_is_the_primary_word_not_the_menu_word(frame):
    model = Sheets()
    r = Refiner(model, conf=1.0, mode="collage", names=MENU, samples=2, every=0)
    for _ in range(2):
        r.run(person(names={0: "person"}), img=frame)
    assert model.calls[0]["hints"] == ["person"]


def test_collage_cell_and_columns_set_the_sheet_size(frame):
    model = Sheets()
    r = Refiner(model, conf=1.0, mode="collage", names=MENU,
                samples=4, every=0, cell=(40, 80), cols=4, )
    for _ in range(4):
        r.run(person(), img=frame)
    # one row of four 40 x 80 cells, with a 4px gutter around and between
    assert model.calls[0]["shape"] == (80 + 2 * 4, 4 * 40 + 5 * 4, 3)


def test_collage_forgets_its_buffer_on_reset(frame):
    model = Sheets()
    r = Refiner(model, conf=1.0, mode="collage", names=MENU, samples=3, every=0)
    r.run(person(), img=frame)
    assert r.shots
    r.reset()
    assert not r.shots and r.frame == -1


def test_collage_caps_how_many_tracks_it_gathers_at_once(frame):
    r = Refiner(Sheets(), conf=1.0, mode="collage", names=MENU, samples=9, every=0, size=2)
    r.run(Tracks(boxes(
        [10, 10, 50, 90, 0.9, 0, 1],
        [10, 10, 50, 90, 0.9, 0, 2],
        [10, 10, 50, 90, 0.9, 0, 3],
    )), img=frame)
    assert len(r.shots) == 2  # the oldest track's tiles are dropped, not kept forever


def test_collage_falls_back_to_batch_for_a_model_that_only_has_name(frame, fixed):
    model = fixed(cls=1)
    r = Refiner(model, conf=1.0, mode="collage", names=MENU, samples=2, every=0)
    r.run(person(), img=frame)
    out = r.run(person(), img=frame)
    assert model.calls == 1        # one name() call, on the collage
    assert out.cls.tolist() == [1]


def test_collage_runs_off_the_frame_loop_under_workers(frame, names):
    model = Gated(cls=1)
    r = Refiner(model, conf=1.0, mode="collage", names=MENU, samples=2, every=0, workers=1)
    try:
        r.run(person(), img=frame)
        out = r.run(person(), img=frame)
        assert out.cls.tolist() == [0]  # still the primary's class
        assert model.entered.wait(5)
        model.gate.set()
        r.wait()
        assert r.run(person(), img=frame).cls.tolist() == [1]
    finally:
        model.gate.set()
        r.close()
