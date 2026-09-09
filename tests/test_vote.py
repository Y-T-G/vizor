from vizor.vote import Vote


def test_majority_wins():
    v = Vote()
    for cls in (2, 7, 7):
        v.add(1, cls)
    assert v.get(1) == 7
    assert v.count(1) == 3


def test_missing_id():
    v = Vote()
    assert v.get(9) is None
    assert v.get(9, 3) == 3
    assert v.count(9) == 0


def test_untracked_ids_are_ignored():
    v = Vote()
    v.add(-1, 5)
    assert len(v) == 0
    assert v.get(-1, 2) == 2


def test_lru_evicts_the_oldest():
    v = Vote(size=2)
    v.add(1, 0)
    v.add(2, 0)
    v.get(1)       # touching 1 makes 2 the oldest
    v.add(3, 0)
    assert 1 in v and 3 in v and 2 not in v


def test_history_is_capped():
    v = Vote(hist=3)
    for cls in (1, 1, 1, 2, 2):
        v.add(4, cls)
    assert v.count(4) == 3
    assert v.get(4) == 2  # the first two votes have aged out


def test_drop_and_clear():
    v = Vote()
    v.add(1, 0)
    v.drop(1)
    assert 1 not in v
    v.add(2, 0)
    v.clear()
    assert len(v) == 0
