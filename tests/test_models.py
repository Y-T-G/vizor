import pickle

import numpy as np
import pytest

from vizor.models.base import Model, ids, menu, parse_id
from vizor.models.pkl import Pkl


def test_base_says_which_role_is_missing():
    m = Model()
    for call, word in ((lambda: m.track(None), "primary"),
                       (lambda: m.find(None), "full"),
                       (lambda: m.name(None), "crop")):
        with pytest.raises(NotImplementedError) as e:
            call()
        assert word in str(e.value)


def test_menu_takes_lists_and_dicts():
    assert menu(["a", "b"]) == "0: a\n1: b"
    assert menu({3: "c"}) == "3: c"
    assert menu(None) == ""


def test_ids():
    assert ids(["a", "b"]) == {0, 1}
    assert ids({7: "truck"}) == {7}
    assert ids(None) is None


@pytest.mark.parametrize("text,want", [
    ("7", 7), ("  7.", 7), ("the answer is 2", 2),
    ("-1", None), ("none", None), (None, None), ("", None),
])
def test_parse_id(text, want):
    assert parse_id(text) == want


def test_parse_id_rejects_ids_outside_the_menu():
    assert parse_id("99", {0, 1, 2}) is None
    assert parse_id("2", {0, 1, 2}) == 2


def test_pkl_replays_in_order(tmp_path):
    path = tmp_path / "p.pkl"
    frames = [np.array([[0, 0, 10, 10, 0.5, 1, 2]], np.float32),
              np.array([[1, 1, 11, 11, 0.6, 3, 2]], np.float32)]
    path.write_bytes(pickle.dumps(frames))
    p = Pkl(path)
    assert len(p) == 2
    assert p.track().cls[0] == 1
    assert p.track().cls[0] == 3
    assert len(p.track()) == 0  # past the end
    p.reset()
    assert p.track().cls[0] == 1


def test_pkl_reorders_ultralytics_columns(tmp_path):
    path = tmp_path / "p.pkl"
    # ultralytics order is [x1, y1, x2, y2, id, conf, cls]
    path.write_bytes(pickle.dumps([np.array([[0, 0, 10, 10, 5, 0.9, 2]], np.float32)]))
    t = Pkl(path, cols=Pkl.ULTRALYTICS).track()
    assert t.conf[0] == pytest.approx(0.9)
    assert t.cls[0] == 2 and t.ids[0] == 5


def test_lazy_imports_do_not_need_torch():
    import vizor

    assert "Florence" in dir(vizor)
    with pytest.raises(AttributeError):
        _ = vizor.NoSuchModel
