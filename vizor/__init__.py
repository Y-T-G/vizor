"""Vizor: a fast detector kept honest by a slower, smarter one.

    import vizor as vz

    viz = vz.Vizor(detector, vz.Florence(names=names), conf=0.5)
    for out in viz.run("traffic.mp4", save="out.mp4"):
        print(out)
"""

__version__ = "0.5.0"

from .boxes import Preds, Track, Tracks, iou
from .core import Vizor
from .refine import Refiner
from .utils.image import montage
from .utils.video import Video, Writer
from .vote import Vote

__all__ = [
    "Vizor", "Refiner", "Vote",
    "Track", "Tracks", "Preds", "iou",
    "Video", "Writer", "montage",
    "Model", "VLM", "HF", "Florence", "Pkl",
    "__version__",
]

_MODELS = {"Model", "VLM", "HF", "Florence", "Pkl"}


def __getattr__(name):
    """Import model wrappers on first use, so torch and friends stay optional."""
    if name in _MODELS:
        from . import models

        return getattr(models, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
