"""Local vision-language models from Hugging Face."""

from typing import Any

import numpy as np

from ..boxes import Preds
from .base import GRID, PROMPT, Model, ids, menu, parse_id

__all__ = ["HF", "Florence"]


def _pil(img):
    """BGR array to a PIL RGB image, which is what the processors expect."""
    from PIL import Image

    if img.ndim == 2:
        return Image.fromarray(img).convert("RGB")
    return Image.fromarray(np.ascontiguousarray(img[..., ::-1]))


class HF(Model):
    """A chat-style VLM loaded locally with transformers.

    Like the hosted models it classifies crops but does not localise, so use it
    with ``mode="crop"``.

    Args:
        model: hub id or local path.
        device: ``"cuda"``, ``"cpu"``, or None to pick whatever is available.
        dtype: torch dtype, defaults to bfloat16 on GPU and float32 on CPU.
        prompt: format string overriding the default. Given ``hint`` and ``menu``.
        grid: format string overriding the collage prompt, used by ``mode="collage"``.
            Given ``n`` (tiles in the collage), ``hint`` and ``menu``.
        gen: generation keyword arguments, e.g. ``max_new_tokens``.
    """

    kind = "chat"

    def __init__(
        self,
        model: str,
        device: "str | None" = None,
        dtype: "str | None" = None,
        prompt: "str | None" = None,
        grid: "str | None" = None,
        gen: "dict[str, Any] | None" = None,
        **kw: Any,
    ):
        import torch
        from transformers import AutoProcessor

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or (torch.bfloat16 if self.device == "cuda" else torch.float32)
        self.prompt = prompt or PROMPT
        self.grid_prompt = grid or GRID
        self.gen = {"max_new_tokens": 16, "do_sample": False, **(gen or {})}
        self.processor = AutoProcessor.from_pretrained(model, trust_remote_code=True)
        self.model = self._load(model, **kw)

    def _auto(self):
        import transformers

        if self.kind == "causallm":
            return transformers.AutoModelForCausalLM
        for cls in ("AutoModelForImageTextToText", "AutoModelForVision2Seq"):
            if hasattr(transformers, cls):
                return getattr(transformers, cls)
        raise ImportError("transformers is too old for image-text models")

    def _load(self, model, **kw):
        """Load with flash attention if it is installed, otherwise plain eager."""
        auto = self._auto()
        opts = dict(torch_dtype=self.dtype, trust_remote_code=True, **kw)
        if self.device == "cuda":
            try:
                return auto.from_pretrained(
                    model, attn_implementation="flash_attention_2", **opts).to(self.device)
            except Exception:
                pass  # flash attention is not installed or the model does not support it
        return auto.from_pretrained(model, attn_implementation="eager", **opts).to(self.device)

    def _text(self, question):
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": question}]}]
        return self.processor.apply_chat_template(messages, add_generation_prompt=True)

    def ask(self, img, question):
        """Send one image and one question, return the decoded reply."""
        inputs = self.processor(text=self._text(question), images=[_pil(img)],
                                return_tensors="pt").to(self.device, self.dtype)
        out = self.model.generate(**inputs, **self.gen)
        out = out[:, inputs["input_ids"].shape[-1]:]  # drop the echoed prompt
        return self.processor.batch_decode(out, skip_special_tokens=True)[0].strip()

    def name(self, crop, names=None, hint=None):
        """Ask the model which class the crop is. Returns a class id, or None if unsure."""
        names = names if names is not None else self.names
        return parse_id(self.ask(crop, self.prompt.format(hint=hint, menu=menu(names))),
                        ids(names))

    def grid(self, collages, names=None, hints=None, tiles=1):
        """Ask about each collage in its own forward pass, with the grid prompt."""
        names = names if names is not None else self.names
        hints = list(hints) if hints is not None else [None] * len(collages)
        valid, lines = ids(names), menu(names)
        return [parse_id(self.ask(s, self.grid_prompt.format(n=tiles, hint=h, menu=lines)), valid)
                for s, h in zip(collages, hints)]


class Florence(HF):
    """Microsoft Florence-2, used as an open-vocabulary detector.

    Florence grounds phrases to boxes, so it works in ``mode="full"``: it labels
    the whole frame in one pass and the refiner matches those boxes to the
    tracks by IoU. It gives no confidence, so every box comes back at 1.0.

    Args:
        model: hub id, e.g. ``"microsoft/Florence-2-base-ft"``.
        names: class id to name mapping. Labels outside it are dropped.
        task: task token. ``"<CAPTION_TO_PHRASE_GROUNDING>"`` looks for the classes
            you list, ``"<OD>"`` reports whatever it finds.
    """

    kind = "causallm"

    def __init__(
        self,
        model: str = "microsoft/Florence-2-base-ft",
        names: "dict[int, str] | list[str] | None" = None,
        task: str = "<CAPTION_TO_PHRASE_GROUNDING>",
        **kw: Any,
    ):
        kw.setdefault("gen", {"max_new_tokens": 1024, "num_beams": 3, "do_sample": False})
        super().__init__(model, **kw)
        self.names = names
        self.task = task

    def _lookup(self, names):
        """Name to id, lowercased, so 'Car' and 'car' both map to the same id."""
        names = names if names is not None else self.names
        if names is None:
            return {}
        items = names.items() if isinstance(names, dict) else enumerate(names)
        return {str(v).lower().strip(): int(k) for k, v in items}

    def run(self, img, task, text=""):
        """Run one Florence task and return its parsed dict."""
        pil = _pil(img)
        inputs = self.processor(text=task + text, images=pil, return_tensors="pt")
        inputs = inputs.to(self.device, self.dtype)
        out = self.model.generate(input_ids=inputs["input_ids"],
                                  pixel_values=inputs["pixel_values"], **self.gen)
        raw = self.processor.batch_decode(out, skip_special_tokens=False)[0]
        return self.processor.post_process_generation(raw, task=task, image_size=pil.size)

    def find(self, img, names=None):
        """Ground the class names in a BGR frame. Every box comes back at confidence 1.0."""
        lookup = self._lookup(names)
        task = self.task
        text = ", ".join(lookup) if task == "<CAPTION_TO_PHRASE_GROUNDING>" and lookup else ""
        out = self.run(img, task, text).get(task, {})
        rows = []
        for box, label in zip(out.get("bboxes", []), out.get("labels", [])):
            cls = lookup.get(str(label).lower().strip())
            if cls is None:
                continue
            rows.append([*box, 1.0, cls, -1])
        data = np.array(rows, np.float32) if rows else np.zeros((0, 7), np.float32)
        return Preds(data, names=names if names is not None else self.names)

    def name(self, crop, names=None, hint=None):
        """Class of the largest thing Florence finds in the crop."""
        preds = self.find(crop, names)
        if not len(preds):
            return None
        areas = (preds.boxes[:, 2] - preds.boxes[:, 0]) * (preds.boxes[:, 3] - preds.boxes[:, 1])
        return int(preds.cls[int(areas.argmax())])

    def grid(self, collages, names=None, hints=None, tiles=1):
        """Largest thing Florence grounds in each collage. It has no chat prompt to word."""
        return [self.name(sheet, names) for sheet in collages]
