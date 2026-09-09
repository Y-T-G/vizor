# Models

A model fills one of three roles. A primary implements `track`, a full mode
secondary implements `find`, a crop mode secondary implements `name`. Some models
fill more than one. `batch` is optional everywhere, and answers about several
crops at once.

| Model | Extra | `track` | `find` | `name` | `batch` |
| --- | --- | --- | --- | --- | --- |
| [`Florence`][vizor.models.hf.Florence] | `hf` | no | yes | yes | one call each |
| [`HF`][vizor.models.hf.HF] | `hf` | no | no | yes | one call each |
| [`VLM`][vizor.models.api.VLM] | `api` | no | no | yes | one request per `chunk` |
| [`Pkl`][vizor.models.pkl.Pkl] | none | yes | yes | no | no |

Anything you leave out raises on the first frame with a message naming the
missing role, rather than silently doing nothing. `batch` is the exception, since
[`Model`][vizor.models.base.Model] gives it a default that calls `name` once per
crop, so a model that never heard of batching still works.

Nothing in that table implements `track` except `Pkl`, which only replays. Bring
your own primary, or copy the ultralytics adapter in `examples/yolo.py`.

## VLM

Talks to any OpenAI-compatible chat completions endpoint. It sends the crop as a
base64 JPEG data url along with a numbered menu of your class names, and reads a
single integer back.

```python
import vizor as vz

# Gemini through its OpenAI compatibility layer. The base url is built in, so
# this needs nothing but GEMINI_API_KEY.
gemini = vz.VLM("gemini-3.1-flash-lite", api="gemini")

# reads the key from OPENAI_API_KEY
openai = vz.VLM("gpt-4o-mini")

# url= points at any OpenAI-compatible server, such as vLLM or llama.cpp. Those
# usually ignore the key, but the client still needs one, hence key="none".
local = vz.VLM("qwen2.5-vl-7b", url="http://localhost:8000/v1", key="none")
```

The key defaults to the provider's variable, `OPENAI_API_KEY` or
`GEMINI_API_KEY`, and constructing the model raises
`ValueError` when there is neither a key nor a url. Never pass a real key as a
literal in code you commit. Put it in the environment.

### Batching

`VLM` puts several crops in one request. `chunk` is how many, and the default is
8.

```python
# up to 8 crops per request, which is the default
viz = vz.Vizor(primary, vz.VLM("gemini-3.1-flash-lite", api="gemini"), mode="crop")

# one crop per request, the way it worked before batching existed
viz = vz.Vizor(primary, vz.VLM("gpt-4o-mini", chunk=1), mode="crop")
```

A batched request numbers each crop and asks for one class id per crop, in
order. The reply is parsed into exactly `chunk` answers. If the model returns too
few, the crops it missed get no vote. If it returns too many, the extras are
dropped. Neither case shifts an answer onto the wrong crop.

Set `chunk=1` if you would rather pay for the round trips than trust a model to
keep eight crops in order.

## Florence

Florence-2 as an open-vocabulary detector. It runs
`<CAPTION_TO_PHRASE_GROUNDING>` with your class names as the prompt and maps the
returned labels back to class ids, case insensitively.

```python
import vizor as vz
from yolo import YOLO

# these names are the grounding prompt, so Florence returns boxes for them only
names = {0: "person", 2: "car", 7: "truck"}
viz = vz.Vizor(YOLO("yolo11n.pt"), vz.Florence(names=names), mode="full")
```

Florence needs the names up front because the prompt is the class list. Passing
`names` to `Vizor` instead works too, and the refiner forwards them.

## Pkl

Replays predictions you dumped earlier, so you can iterate on the refiner with no
model loaded and no API calls.

```python
import vizor as vz

# saved ultralytics output, reindexed into [x1, y1, x2, y2, conf, cls, id]
primary = vz.Pkl("tracks.pkl", cols=vz.Pkl.ULTRALYTICS)

# already in vizor order, so no reindex
secondary = vz.Pkl("florence.pkl")

# each call hands out the next saved frame, so the two files have to line up
# with each other and with the video they came from
viz = vz.Vizor(primary, secondary, conf=0.5, mode="full")
```

`cols` reindexes each row into the `[x1, y1, x2, y2, conf, cls, id]` layout.
`Pkl.ULTRALYTICS` is `[0, 1, 2, 3, 5, 6, 4]`, which is what ultralytics gives you
for tracked boxes. It is only a column order, so `Pkl` needs nothing installed.

## Writing your own

Subclass [`Model`][vizor.models.base.Model] and implement the method your role
needs.

```python
import numpy as np
from vizor import Model, Tracks

class MyDetector(Model):
    # turns class ids into labels when drawing, and becomes the menu the VLM picks from
    names = {0: "person", 1: "car"}

    def track(self, img):
        # img is BGR, the layout OpenCV hands you. Return one row per object as
        # [x1, y1, x2, y2, conf, cls, id], with id = -1 for anything untracked.
        # An untracked row is refined on this frame and never cached.
        return Tracks(np.zeros((0, 7), np.float32), names=self.names)
```

Images handed to your model are BGR, the layout OpenCV gives you. Convert inside
your wrapper if the model wants RGB.

`name` returns a class id, or `None` when the model is not sure. `None` records no
vote, so an unsure answer leaves the primary's class in place rather than
overwriting it with a guess.

Two helpers in [`vizor.models.base`](reference/models.md) do the prompt work for
you if you are wrapping a chat model. `menu(names)` renders the class list as
`id: name` lines, and `parse_id(text, valid)` pulls the first integer out of a
reply and returns `None` when it is negative or outside the menu.
