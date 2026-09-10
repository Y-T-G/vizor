"""What Vizor expects a model to look like."""

from ..boxes import Preds, Tracks  # noqa: F401  (re-exported for implementers)

__all__ = ["Model", "PROMPT", "BATCH", "GRID", "menu", "ids", "parse_id", "parse_ids"]


class Model:
    """Base class for both roles.

    A primary implements ``track``. A secondary implements ``find``, ``name``,
    or both, depending on which refiner mode it is meant to serve. Anything not
    implemented raises, so a mismatch between model and mode fails loudly on the
    first frame instead of silently doing nothing.

    ``names`` is the class id to name mapping the model works in. Leave it None
    if the model has no fixed vocabulary.
    """

    names = None

    def track(self, img):
        """Detect and track on a BGR frame. Returns [Tracks][vizor.boxes.Tracks]."""
        raise NotImplementedError(f"{type(self).__name__} cannot be a primary model")

    def find(self, img, names=None):
        """Detect on a whole BGR frame. Returns [Preds][vizor.boxes.Preds]."""
        raise NotImplementedError(f"{type(self).__name__} does not support mode='full'")

    def name(self, crop, names=None, hint=None):
        """Classify a BGR crop. Returns a class id, or None if unsure.

        ``hint`` is what the primary thought the object was.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support mode='crop'")

    def batch(self, crops, names=None, hints=None):
        """Classify several BGR crops. Returns one class id or None per crop.

        The default asks ``name`` once per crop, so a model that only implements
        ``name`` works unchanged. Override this when the model can do the whole
        list in one call, which is what makes crop mode cheap.

        ``hints`` is what the primary thought each object was, in the same order.
        """
        hints = list(hints) if hints is not None else [None] * len(crops)
        return [self.name(c, names, h) for c, h in zip(crops, hints)]

    def grid(self, collages, names=None, hints=None, tiles=1):
        """Classify collages, each one object shown ``tiles`` times over a video.

        Returns one class id or None per collage, the same shape as ``batch``.
        The default forwards to ``batch``, which treats a collage as an ordinary
        image, so a model that has never heard of collages still answers.
        Override it to word the prompt for a grid.
        """
        return self.batch(collages, names, hints)

    def reset(self):
        """Drop any per-video state. Called by ``Vizor.reset``."""

    def __repr__(self):
        return f"{type(self).__name__}({getattr(self, 'model', '')!r})"


PROMPT = (
    "This is a crop of one object taken from a detector's box. "
    "The detector called it {hint!r}, which may be wrong. "
    "Which of these classes is it?\n{menu}\n"
    "Reply with the class id only: a single integer, no words, no punctuation. "
    "Reply -1 if none of them fit."
)


BATCH = (
    "These are {n} crops, each one object taken from a detector's box, "
    "given in order and numbered.\n"
    "The detector guessed: {hints}. Those guesses may be wrong.\n"
    "Which of these classes is each crop?\n{menu}\n"
    "Reply with exactly {n} class ids, one per crop, in the same order, "
    "separated by commas. Use -1 for a crop where none of them fit. "
    "No words, no explanation, only the numbers and commas."
)


GRID = (
    "This is a grid of {n} crops of the same object, cut from a detector's box "
    "at different moments in a video and laid out left to right, top to bottom.\n"
    "The detector called it {hint!r}, which may be wrong.\n"
    "Some crops may be blurred, partly hidden or badly lit. Weigh them together "
    "and answer once for the object, not once per crop.\n"
    "Which of these classes is it?\n{menu}\n"
    "Reply with the class id only: a single integer, no words, no punctuation. "
    "Reply -1 if none of them fit."
)


def menu(names, limit=200):
    """Render a class mapping as ``id: name`` lines for a prompt."""
    if names is None:
        return ""
    items = names.items() if isinstance(names, dict) else enumerate(names)
    items = list(items)[:limit]
    return "\n".join(f"{int(k)}: {v}" for k, v in items)


def ids(names):
    """The set of valid class ids in a mapping, or None if there is no mapping."""
    if names is None:
        return None
    return {int(k) for k in (names.keys() if isinstance(names, dict) else range(len(names)))}


def parse_id(text, valid=None):
    """Pull a class id out of a model's reply. None if there isn't a usable one."""
    import re

    if text is None:
        return None
    found = re.search(r"-?\d+", str(text))
    return _check(found.group(), valid) if found else None


def parse_ids(text, n, valid=None):
    """Pull ``n`` class ids out of a batched reply, in order.

    A reply with too few numbers is padded with None, one with too many is cut.
    Either way the caller gets exactly ``n`` entries, so a model that miscounts
    costs some crops their refinement rather than shifting every later answer
    onto the wrong object.
    """
    import re

    found = re.findall(r"-?\d+", str(text)) if text is not None else []
    out = [_check(x, valid) for x in found[:n]]
    return out + [None] * (n - len(out))


def _check(text, valid):
    """A parsed id, or None when it is negative or outside the menu."""
    cls = int(text)
    if cls < 0:
        return None
    if valid is not None and cls not in valid:
        return None
    return cls
