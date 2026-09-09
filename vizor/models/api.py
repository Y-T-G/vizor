"""Chat VLMs behind an OpenAI-style HTTP API."""

import os

import cv2

from .base import PROMPT, Model, ids, menu, parse_id

__all__ = ["Vlm"]

ENV = {"openai": "OPENAI_API_KEY", "groq": "GROQ_API_KEY"}


class Vlm(Model):
    """A hosted vision model asked to classify one crop at a time.

    These models describe an image well but do not give boxes, so use them with
    ``mode="crop"``. The primary keeps the box, the VLM only fixes the class.

    Args:
        model: model id, e.g. ``"gpt-4o-mini"`` or ``"llama-3.2-11b-vision-preview"``.
        api: ``"openai"``, ``"groq"``, or ``"custom"`` with a ``url``.
        key: API key. Read from ``OPENAI_API_KEY`` or ``GROQ_API_KEY`` if left out.
        url: base url, for self-hosted or third party OpenAI-compatible servers.
        prompt: format string overriding the default. It is given ``hint`` and ``menu``.
        kw: forwarded to the chat completion call, e.g. ``temperature``, ``max_tokens``.

    Never pass a key as a literal in code you commit. Put it in the environment.
    """

    def __init__(self, model, api="openai", key=None, url=None, prompt=None, **kw):
        self.model = model
        self.api = api
        self.prompt = prompt or PROMPT
        self.kw = {"temperature": 0, "max_tokens": 16, **kw}
        key = key or os.environ.get(ENV.get(api, ""))
        if key is None and url is None:
            raise ValueError(
                f"no API key: pass key= or set {ENV.get(api, 'the provider env var')}"
            )
        self.client = self._client(api, key, url)

    @staticmethod
    def _client(api, key, url):
        opts = {"api_key": key}
        if url:
            opts["base_url"] = url
        if api == "groq":
            from groq import Groq

            return Groq(**opts)
        if api in ("openai", "custom"):
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

    def ask(self, img, text):
        """Send one image and one question, return the reply as a string."""
        reply = self.client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": self.encode(img)}},
                    {"type": "text", "text": text},
                ],
            }],
            **self.kw,
        )
        return reply.choices[0].message.content

    def name(self, crop, names=None, hint=None):
        names = names if names is not None else self.names
        text = self.prompt.format(hint=hint, menu=menu(names))
        return parse_id(self.ask(crop, text), ids(names))
