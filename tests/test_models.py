import pickle
import types

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


# batching --------------------------------------------------------------


def test_parse_ids_pads_and_truncates():
    from vizor.models.base import parse_ids

    assert parse_ids("2, 5, 7", 3) == [2, 5, 7]
    assert parse_ids("2, 5", 3) == [2, 5, None]      # short reply loses the last crop
    assert parse_ids("1 2 3 4", 2) == [1, 2]         # long reply is cut, not shifted
    assert parse_ids(None, 2) == [None, None]


def test_parse_ids_drops_ids_outside_the_menu():
    from vizor.models.base import parse_ids

    assert parse_ids("2, 99, -1", 3, {2, 7}) == [2, None, None]


def test_base_batch_falls_back_to_one_name_call_each():
    class Loop(Model):
        def __init__(self):
            self.seen = []

        def name(self, crop, names=None, hint=None):
            self.seen.append((crop, hint))
            return len(self.seen)

    m = Loop()
    assert m.batch(["a", "b", "c"], hints=["x", "y", "z"]) == [1, 2, 3]
    assert m.seen == [("a", "x"), ("b", "y"), ("c", "z")]
    assert Loop().batch(["a"]) == [1]  # hints are optional


class FakeChat:
    """Stands in for client.chat.completions, recording every request."""

    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    @property
    def chat(self):
        return self

    @property
    def completions(self):
        return self

    def create(self, **kw):
        self.calls.append(kw)
        text = self.reply(len(self.calls)) if callable(self.reply) else self.reply
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=text))]
        )


def make_vlm(monkeypatch, reply, seen=None, **kw):
    from vizor.models.api import VLM

    fake = FakeChat(reply)

    def client(api, key, url):
        if seen is not None:
            seen.update(api=api, key=key, url=url)
        return fake

    monkeypatch.setattr(VLM, "_client", staticmethod(client))
    return VLM("m", key="k", **kw), fake


def crops(n):
    return [np.full((8, 8, 3), i, np.uint8) for i in range(1, n + 1)]


def test_vlm_sends_one_request_for_a_whole_chunk(monkeypatch):
    m, fake = make_vlm(monkeypatch, "2, 7, 0", chunk=8)
    assert m.batch(crops(3), {0: "person", 2: "car", 7: "truck"}) == [2, 7, 0]
    assert len(fake.calls) == 1
    content = fake.calls[0]["messages"][0]["content"]
    assert sum(c["type"] == "image_url" for c in content) == 3
    assert "Crop 1:" in [c.get("text") for c in content]
    assert fake.calls[0]["max_tokens"] >= 24  # 16 would not hold three ids


def test_vlm_splits_into_chunks(monkeypatch):
    m, fake = make_vlm(monkeypatch, "1, 1", chunk=2)
    assert m.batch(crops(4), {1: "bicycle"}) == [1, 1, 1, 1]
    assert len(fake.calls) == 2


def test_vlm_chunk_of_one_sends_them_one_at_a_time(monkeypatch):
    m, fake = make_vlm(monkeypatch, "7", chunk=1)
    assert m.batch(crops(3), {7: "truck"}) == [7, 7, 7]
    assert len(fake.calls) == 3
    # a lone crop uses the single-crop prompt, so it is not numbered
    content = fake.calls[0]["messages"][0]["content"]
    assert "Crop 1:" not in [c.get("text") for c in content]


def test_vlm_short_reply_loses_only_the_crops_it_missed(monkeypatch):
    m, _ = make_vlm(monkeypatch, "2", chunk=8)
    assert m.batch(crops(3), {2: "car"}) == [2, None, None]


def test_vlm_gemini_gets_the_google_url_and_env_var(monkeypatch):
    from vizor.models.api import ENV, URL

    seen = {}
    monkeypatch.setenv("GEMINI_API_KEY", "from-env")
    from vizor.models.api import VLM

    fake = FakeChat("0")
    monkeypatch.setattr(
        VLM, "_client",
        staticmethod(lambda api, key, url: (seen.update(api=api, key=key, url=url), fake)[1]),
    )
    VLM("gemini-2.0-flash", api="gemini")
    assert seen["url"] == URL["gemini"] == \
        "https://generativelanguage.googleapis.com/v1beta/openai/"
    assert seen["key"] == "from-env"
    assert ENV["gemini"] == "GEMINI_API_KEY"


def test_vlm_without_a_key_says_which_variable_to_set(monkeypatch):
    from vizor.models.api import VLM

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ValueError) as e:
        VLM("gemini-2.0-flash", api="gemini")
    assert "GEMINI_API_KEY" in str(e.value)
