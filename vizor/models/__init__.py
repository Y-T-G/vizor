"""Model wrappers. Each one is imported lazily so its dependency stays optional."""

from .base import Model

__all__ = ["Model", "Yolo", "Vlm", "Hf", "Florence", "Pkl"]

_WHERE = {"Yolo": "yolo", "Vlm": "api", "Hf": "hf", "Florence": "hf", "Pkl": "pkl"}


def __getattr__(name):
    if name in _WHERE:
        import importlib

        return getattr(importlib.import_module(f".{_WHERE[name]}", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
