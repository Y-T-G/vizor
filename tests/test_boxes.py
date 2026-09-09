import numpy as np
import pytest
from conftest import boxes

from vizor.boxes import Preds, Track, Tracks, iou


def test_empty():
    t = Tracks()
    assert len(t) == 0 and not t
    assert t.data.shape == (0, 7)
    assert list(t) == []


def test_six_columns_get_an_id():
    p = Preds(np.array([[0, 0, 10, 10, 0.9, 2]], np.float32))
    assert p.data.shape == (1, 7)
    assert p[0].id == -1


def test_bad_width():
    with pytest.raises(ValueError):
        Tracks(np.zeros((2, 5), np.float32))


def test_index_and_write_back():
    t = Tracks(boxes([0, 0, 10, 10, 0.4, 0, 3]))
    one = t[0]
    assert isinstance(one, Track) and one.id == 3 and one.cls == 0
    one.cls = 7
    t[0] = one
    assert t.cls[0] == 7


def test_slice_is_a_copy():
    t = Tracks(boxes([0, 0, 10, 10, 0.4, 0, 3], [5, 5, 9, 9, 0.9, 1, 4]))
    sub = t[t.conf > 0.5]
    assert isinstance(sub, Tracks) and len(sub) == 1 and sub.ids[0] == 4
    sub.data[0, 5] = 2
    assert t.cls[1] == 1  # the original is untouched


def test_columns():
    t = Tracks(boxes([0, 0, 10, 20, 0.4, 2, 3]))
    assert t.boxes.tolist() == [[0, 0, 10, 20]]
    assert t.conf.tolist() == [pytest.approx(0.4)]
    assert t.cls.tolist() == [2] and t.ids.tolist() == [3]
    assert t[0].wh == (10, 20) and t[0].area == 200


def test_names(names):
    t = Tracks(boxes([0, 0, 10, 10, 0.4, 2, 3]), names=names)
    assert t.name(2) == "car"
    assert t.name(99) == "99"  # missing ids fall back to the number
    assert Tracks(boxes([0, 0, 1, 1, 0.1, 5, 1])).name(5) == "5"


def test_draw(frame, names):
    t = Tracks(boxes([10, 10, 100, 200, 0.42, 2, 3]), names=names, img=frame)
    out = t.draw()
    assert out.shape == frame.shape
    assert out is frame  # drawing happens in place
    assert (out != 32).any()


def test_draw_without_image():
    with pytest.raises(ValueError):
        Tracks(boxes([0, 0, 1, 1, 0.5, 0, 1])).draw()


def test_iou():
    a = np.array([[0, 0, 10, 10]], np.float32)
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]], np.float32)
    m = iou(a, b)
    assert m.shape == (1, 3)
    assert m[0, 0] == pytest.approx(1.0)
    assert m[0, 1] == pytest.approx(1 / 3)
    assert m[0, 2] == pytest.approx(0.0)


def test_iou_empty():
    assert iou(np.zeros((0, 4), np.float32), np.zeros((3, 4), np.float32)).shape == (0, 3)
