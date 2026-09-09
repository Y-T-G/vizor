"""Per-track class votes.

The secondary model is slow, so we do not want to ask it about the same track
on every frame. Each answer is stored as a vote against the track id and the
running majority is reused for free on later frames. Ids are held in an LRU so
a long video cannot grow the cache without bound.
"""

from collections import Counter, OrderedDict, deque

__all__ = ["Vote"]


class Vote:
    """Majority vote per track id, capped at ``size`` ids and ``hist`` votes each.

    Ids below zero are untracked detections that share the same placeholder id,
    so they are ignored instead of being pooled together.
    """

    def __init__(self, size=256, hist=25):
        self.size = int(size)
        self.hist = int(hist)
        self.data = OrderedDict()

    def add(self, id, cls):
        """Record one vote of ``cls`` for track ``id``."""
        id = int(id)
        if id < 0:
            return
        dq = self.data.get(id)
        if dq is None:
            dq = self.data[id] = deque(maxlen=self.hist)
            while len(self.data) > self.size:
                self.data.popitem(last=False)
        self.data.move_to_end(id)
        dq.append(int(cls))

    def get(self, id, default=None):
        """Most common class voted for ``id``, or ``default`` if it has no votes."""
        id = int(id)
        dq = self.data.get(id) if id >= 0 else None
        if not dq:
            return default
        self.data.move_to_end(id)
        return Counter(dq).most_common(1)[0][0]

    def count(self, id):
        """How many votes ``id`` has, capped at ``hist``."""
        dq = self.data.get(int(id))
        return len(dq) if dq else 0

    def drop(self, id):
        """Forget one track id. Does nothing if it was never seen."""
        self.data.pop(int(id), None)

    def clear(self):
        """Forget every track id."""
        self.data.clear()

    def __contains__(self, id):
        return int(id) in self.data

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return f"Vote({len(self)}/{self.size} ids, hist={self.hist})"
