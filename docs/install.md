# Install

The core needs numpy and OpenCV only. Everything else is an optional extra, so
you install the one your models actually need.

```sh
pip install vizor
```

That gives you [`Vizor`][vizor.core.Vizor], [`Refiner`][vizor.refine.Refiner],
[`Tracks`][vizor.boxes.Tracks], [`Vote`][vizor.vote.Vote] and the video helpers.
It does not give you any model, and it never gives you a detector. See
[licensing](models.md#licensing) for why the detector is yours to bring.

## Extras

Each extra pulls in the dependency for one group of model wrappers.

| Extra | Installs | Gets you |
| --- | --- | --- |
| `hf` | `torch`, `transformers`, `pillow`, `einops`, `timm` | [`HF`][vizor.models.hf.HF] and [`Florence`][vizor.models.hf.Florence] |
| `api` | `openai` | [`VLM`][vizor.models.api.VLM] against OpenAI or any compatible url |
| `groq` | `groq` | `VLM(api="groq")` |
| `dev` | `pytest`, `ruff` | the test suite and the linter |
| `docs` | `zensical`, `mkdocstrings[python]` | building this site |

Combine them with a comma.

```sh
pip install "vizor[api,hf]"
```

[`Pkl`][vizor.models.pkl.Pkl] needs no extra. It replays predictions you saved
earlier, so it works on the core install.

## From source

Clone and install in editable mode if you want to change the package or run the
tests.

```sh
git clone https://github.com/Y-T-G/vizor
cd vizor
pip install -e ".[dev]"
pytest
```

The suite needs no GPU, no API key and no model weights. It stubs the secondary
instead of calling one.

```
59 passed in 3.65s
```

The replay tests in `tests/test_replay.py` skip themselves when the cached
predictions are not on disk, and those files are not in the repo, so a fresh
clone reports fewer than 59.

## Lazy imports

`import vizor` never imports torch, transformers, openai or groq. The model
wrappers are resolved through a module-level `__getattr__` and imported the first
time you name one.

```python
import vizor as vz   # numpy and OpenCV only
model = vz.Florence()  # transformers and torch are imported here
```

The practical consequence is that a missing extra shows up as an `ImportError`
when you construct the model, not when you import the package.
