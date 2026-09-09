"""Chat VLMs behind an OpenAI-style HTTP API."""

import os
from typing import Any

import cv2

from .base import BATCH, PROMPT, Model, ids, menu, parse_id, parse_ids

__all__ = ["VLM"]

ENV = {
    "openai": "OPENAI_API_KEY",
    "groq": "GROQ_API_KEY",
    "gemini": "GEMINI_API_KEY",
}

# providers that speak the OpenAI protocol but live somewhere else
URL = {"gemini": "https://generativelanguage.googleapis.com/v1beta/openai/"}


class VLM(Model):
    """A hosted vision model asked which class a crop is.

    These models describe an image well but do not give boxes, so use them with
    ``mode="crop"``. The primary keeps the box, the VLM only fixes the class.

    Args:
        model: model id, e.g. ``"gpt-4o-mini"`` or ``"gemini-2.0-flash"``.
        api: ``"openai"``, ``"groq"``, ``"gemini"``, or ``"custom"`` with a ``url``.
        key: API key. Read from the provider's env var if left out, one of
            ``OPENAI_API_KEY``, ``GROQ_API_KEY`` or ``GEMINI_API_KEY``.
        url: base url, for self-hosted or third party OpenAI-compatible servers.
            Gemini has one built in, so ``api="gemini"`` needs no url.
        prompt: format string overriding the default. It is given ``hint`` and ``menu``.
        chunk: how many crops go in one request. 1 sends them one at a time.
        batch: format string overriding the batched prompt. It is given ``n``,
            ``hints`` and ``menu``.
        kw: forwarded to the chat completion call, e.g. ``temperature``, ``max_tokens``.

    Never pass a key as a literal in code you commit. Put it in the environment.
    """

    def __init__(
        self,
        model: str,
        api: str = "openai",
        key: "str | None" = None,
        url: "str | None" = None,
        prompt: "str | None" = None,
        chunk: int = 8,
        batch: "str | None" = None,
        **kw: Any,
    ):
        self.model = model
        self.api = api
        self.prompt = prompt or PROMPT
        self.batch_prompt = batch or BATCH
        self.chunk = max(1, int(chunk))
        self.kw = {"temperature": 0, "max_tokens": 16, **kw}
        key = key or os.environ.get(ENV.get(api, ""))
        if key is None and url is None:
            raise ValueError(
                f"no API key: pass key= or set {ENV.get(api, 'the provider env var')}"
            )
        self.client = self._client(api, key, url or URL.get(api))

    @staticmethod
    def _client(api, key, url):
        opts = {"api_key": key}
        if url:
            opts["base_url"] = url
        if api == "groq":
            from groq import Groq

            return Groq(**opts)
        if api in ("openai", "gemini", "custom"):
            from openai import OpenAI

            return OpenAI(**opts)
        raise ValueError(f"unknown api {api!r}, expected one of {sorted(ENV) + ['custom']}")

    @staticmethod
    def encode(img, quality=90):
        """BGR array to a base64 data url the chat APIs accept."""
        import base64

        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise ValueError("could not encode crop as jpeg")
        return "data:image/jpeg;base64," + base64.b64encode(buf).decode()

    def ask(self, imgs, text, **kw):
        """Send one or more images with one question, return the reply as a string.

        Several images go in a single message, each preceded by its number, so
        the model can answer about all of them at once.
        """
        if not isinstance(imgs, (list, tuple)):
            imgs = [imgs]
        content = []
        for i, img in enumerate(imgs, 1):
            if len(imgs) > 1:
                content.append({"type": "text", "text": f"Crop {i}:"})
            content.append({"type": "image_url", "image_url": {"url": self.encode(img)}})
        content.append({"type": "text", "text": text})
        reply = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            **{**self.kw, **kw},
        )
        return reply.choices[0].message.content

    def name(self, crop, names=None, hint=None):
        """Ask the model which class the crop is. Returns a class id, or None if unsure."""
        names = names if names is not None else self.names
        text = self.prompt.format(hint=hint, menu=menu(names))
        return parse_id(self.ask(crop, text), ids(names))

    def batch(self, crops, names=None, hints=None):
        """Classify several crops, ``chunk`` of them per request.

        One request carrying eight crops costs one round trip instead of eight,
        which is most of the wall clock in crop mode. The trade is that the model
        has to keep the order straight, and a reply with the wrong count loses
        the crops it did not cover rather than shifting the rest.
        """
        names = names if names is not None else self.names
        hints = list(hints) if hints is not None else [None] * len(crops)
        valid, lines = ids(names), menu(names)
        out = []
        for i in range(0, len(crops), self.chunk):
            part, hint = crops[i:i + self.chunk], hints[i:i + self.chunk]
            if len(part) == 1:
                out.append(self.name(part[0], names, hint[0]))
                continue
            text = self.batch_prompt.format(
                n=len(part),
                hints=", ".join(f"{j}. {h!r}" for j, h in enumerate(hint, 1)),
                menu=lines,
            )
            # 16 tokens holds one id, not eight, so give the reply room to fit
            room = max(self.kw.get("max_tokens", 16), 8 * len(part))
            out += parse_ids(self.ask(part, text, max_tokens=room), len(part), valid)
        return out
