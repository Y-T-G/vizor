"""What Vizor expects a model to look like."""

from ..boxes import Preds, Tracks  # noqa: F401  (re-exported for implementers)

__all__ = ["Model", "PROMPT", "menu", "ids", "parse_id"]


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
        """Detect and track on a BGR frame. Returns :class:`~vizor.boxes.Tracks`."""
        raise NotImplementedError(f"{type(self).__name__} cannot be a primary model")

    def find(self, img, names=None):
        """Detect on a whole BGR frame. Returns :class:`~vizor.boxes.Preds`."""
        raise NotImplementedError(f"{type(self).__name__} does not support mode='full'")

    def name(self, crop, names=None, hint=None):
        """Classify a BGR crop. Returns a class id, or None if unsure.

        ``hint`` is what the primary thought the object was.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support mode='crop'")

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
    if not found:
        return None
    cls = int(found.group())
    if cls < 0:
        return None
    if valid is not None and cls not in valid:
        return None
    return cls
