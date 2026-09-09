# Models

A model fills one of three roles. A primary implements `track`, a full mode
secondary implements `find`, a crop mode secondary implements `name`. Some models
fill more than one.

| Model | Extra | `track` | `find` | `name` |
| --- | --- | --- | --- | --- |
| [`Florence`][vizor.models.hf.Florence] | `hf` | no | yes | yes |
| [`HF`][vizor.models.hf.HF] | `hf` | no | no | yes |
| [`VLM`][vizor.models.api.VLM] | `api` or `groq` | no | no | yes |
| [`Pkl`][vizor.models.pkl.Pkl] | none | yes | yes | no |

Anything you leave out raises on the first frame with a message naming the
missing role, rather than silently doing nothing.

Nothing in that table implements `track` except `Pkl`, which only replays. The
package ships no live detector, and [licensing](#licensing) says why. Bring your
own, or copy the ultralytics adapter in `examples/yolo.py`.

## VLM

Talks to any OpenAI-compatible chat completions endpoint. It sends the crop as a
base64 JPEG data url along with a numbered menu of your class names, and reads a
single integer back.

```python
import vizor as vz

groq = vz.VLM("meta-llama/llama-4-scout-17b-16e-instruct", api="groq")
local = vz.VLM("qwen2.5-vl-7b", url="http://localhost:8000/v1", key="none")
```

The key defaults to `OPENAI_API_KEY` or `GROQ_API_KEY` depending on `api`, and
constructing the model raises `ValueError` when neither is set. Never pass a real
key as a literal in code you commit. Put it in the environment.

## Florence

Florence-2 as an open-vocabulary detector. It runs
`<CAPTION_TO_PHRASE_GROUNDING>` with your class names as the prompt and maps the
returned labels back to class ids, case insensitively.

```python
import vizor as vz
from yolo import YOLO

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

primary = vz.Pkl("tracks.pkl", cols=vz.Pkl.ULTRALYTICS)
secondary = vz.Pkl("florence.pkl")
viz = vz.Vizor(primary, secondary, conf=0.5, mode="full")
```

`cols` reindexes each row into the `[x1, y1, x2, y2, conf, cls, id]` layout.
`Pkl.ULTRALYTICS` is `[0, 1, 2, 3, 5, 6, 4]`, which is what ultralytics gives you
for tracked boxes. It is a column order, not ultralytics code, so `Pkl` needs
nothing installed.

## Writing your own

Subclass [`Model`][vizor.models.base.Model] and implement the method your role
needs.

```python
import numpy as np
from vizor import Model, Tracks

class MyDetector(Model):
    names = {0: "person", 1: "car"}

    def track(self, img):
        # your model here, returning [x1, y1, x2, y2, conf, cls, id] rows
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

## Licensing

vizor is Apache-2.0. Ultralytics is AGPL-3.0, and AGPL says a work that combines
with it must also be AGPL-3.0. A Python module that imports ultralytics forms
that combined work when it runs, so shipping a YOLO wrapper inside an Apache-2.0
wheel would put the two licences in conflict.

So the wrapper is not in the package. It is `examples/yolo.py`, marked
`AGPL-3.0-or-later`, and pip never installs it. Nothing that pip installs imports
ultralytics, and you can check that.

```sh
pip download vizor --no-deps -d /tmp/v && unzip -o -q /tmp/v/vizor-*.whl -d /tmp/v
grep -rn "^[[:space:]]*\(import\|from\) ultralytics" /tmp/v/vizor/ ; echo "exit $?"
```

That prints `exit 1` and nothing else, meaning grep matched no line. The name
still appears in `pkl.py`, in a comment and in the `Pkl.ULTRALYTICS` column
order. That is a description of a data layout, not ultralytics code, and it
imports nothing.

What it means for you. If your own project is AGPL-3.0, or you hold an
Ultralytics Enterprise licence, copy `examples/yolo.py` and use it. If your
project is closed source or permissively licensed, write an adapter for a
detector whose licence you can live with. The interface is `track`, above.

I am not a lawyer and this is not legal advice. If the answer matters
commercially, ask one.
