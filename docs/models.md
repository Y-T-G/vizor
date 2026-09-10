# Models

A model fills one of three roles. A primary implements `track`, a full mode
secondary implements `find`, a crop mode secondary implements `name`. Some models
fill more than one. `batch` and `grid` are optional everywhere: `batch` answers
about several crops at once, `grid` about a collage of one track.

| Model | Extra | `track` | `find` | `name` | `batch` | `grid` |
| --- | --- | --- | --- | --- | --- | --- |
| [`Florence`][vizor.models.hf.Florence] | `hf` | no | yes | yes | one call each | one call each |
| [`HF`][vizor.models.hf.HF] | `hf` | no | no | yes | one call each | one call each |
| [`VLM`][vizor.models.api.VLM] | `api` | no | no | yes | one request per `chunk` | one request each |
| [`Pkl`][vizor.models.pkl.Pkl] | none | yes | yes | no | no | no |

Anything you leave out raises on the first frame with a message naming the
missing role, rather than silently doing nothing. `batch` and `grid` are the
exceptions. [`Model`][vizor.models.base.Model] gives `batch` a default that calls
`name` once per crop and `grid` a default that hands the collage to `batch`, so a
model that never heard of either still works.

Nothing in that table implements `track` except `Pkl`, which only replays. Bring
your own primary, or copy the ultralytics adapter in `examples/yolo.py`. That
adapter also works as a secondary, which is the last section here.

Every section below ends in a program you can run start to finish. They all use
`YOLO` from `examples/yolo.py` as the primary, except the `Pkl` one, which needs
no model at all.

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

### End to end

Install the extra and the detector, and put the key in the environment.

```sh
pip install "vizor[api]" ultralytics
export GEMINI_API_KEY=...
```

This tracks a traffic video, asks Gemini about the boxes YOLO scored at or below
0.4, and writes an annotated copy.

```python
import vizor as vz
from yolo import YOLO   # examples/yolo.py

names = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

viz = vz.Vizor(
    YOLO("yolo11n.pt"),
    vz.VLM("gemini-3.1-flash-lite", api="gemini", chunk=8),
    conf=0.4,
    mode="crop",
    names=names,
    workers=2,   # the frame loop does not wait for Gemini
)

with viz:
    for out in viz.run("traffic.mp4", save="out.mp4"):
        print(f"{len(out)} boxes, {len(viz.refiner.cache)} tracks refined", end="\r")
```

The counter on the right is the number of distinct objects that have cost a
request. It climbs far more slowly than the frame count, and that gap is the
whole point of the vote cache.

### Asking about a collage

`VLM.grid` sends one collage per request, using a prompt that says the tiles are
the same object at different moments. Collage mode calls it for you, but it takes
plain images, so you can call it yourself.

```python
import vizor as vz

vlm = vz.VLM("gemini-3.1-flash-lite", api="gemini")
sheet = vz.montage(crops, cols=3, cell=(96, 192))   # six crops of one person
vlm.grid([sheet], {0: "man", 1: "woman"}, ["person"], tiles=len(crops))
```

Override the wording with `grid=`, the same way `prompt=` overrides the single
crop one. It is given `n`, the number of tiles, along with `hint` and `menu`.

```python
ask = ("{n} crops of one person over a few seconds.\n"
       "Which is it?\n{menu}\nReply with the id only.")
vlm = vz.VLM("gemini-3.1-flash-lite", api="gemini", grid=ask)
```

### End to end, a question the detector cannot answer

The class list you hand the refiner does not have to be the detector's. Here YOLO
finds people and Gemini says which of two labels each one gets, from six crops
spread across 40 frames of that person's track.

```sh
pip install "vizor[api]" ultralytics
export GEMINI_API_KEY=...
```

The primary is filtered to class 0, person, so nothing else can come back
labelled man or woman.

```python
import vizor as vz
from yolo import YOLO   # examples/yolo.py

viz = vz.Vizor(
    YOLO("yolo11n.pt", classes=[0]),
    vz.VLM("gemini-3.1-flash-lite", api="gemini"),
    conf=1.0,                  # not about doubt, so every track gets asked
    mode="collage",
    names={0: "man", 1: "woman"},
    samples=6,                 # six crops make one collage
    every=8,                   # taken eight frames apart
    cell=(96, 192),            # tall cells, because people are
    cols=3,
    workers=2,
)

with viz:
    for out in viz.run("street.mp4", save="out.mp4"):
        print(f"{len(out)} people, {len(viz.refiner.cache)} answered, "
              f"{len(viz.refiner.shots)} part way", end="\r")
```

A track carries the detector's label until its collage is full and answered. The
sixth crop is taken five gaps of eight frames after the first, so at these
settings no answer arrives until a track has been in view for 40 frames. The
third counter is how many tracks are still gathering crops.

`python examples/collage.py street.mp4 --save out.mp4` is this same program with
arguments.

## HF

The same idea as `VLM`, but the model runs on your own machine through
transformers. It classifies crops and does not localise, so it belongs in
`mode="crop"`.

```python
import vizor as vz

# picks cuda when torch sees a GPU, else cpu. bfloat16 on GPU, float32 on cpu.
qwen = vz.HF("Qwen/Qwen2.5-VL-3B-Instruct")

# pin the device and give the reply more room
qwen = vz.HF("Qwen/Qwen2.5-VL-3B-Instruct", device="cuda", gen={"max_new_tokens": 32})
```

`HF` has no `batch` of its own, so it inherits the default and runs one crop per
forward pass. It does have `grid`, so collage mode words the prompt properly. It
is a single local model either way, so do not put it behind `workers`.

### End to end

```sh
pip install "vizor[hf]" ultralytics
```

This keeps everything on one machine, no network and no key.

```python
import vizor as vz
from yolo import YOLO   # examples/yolo.py

names = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

viz = vz.Vizor(
    YOLO("yolo11n.pt"),
    vz.HF("Qwen/Qwen2.5-VL-3B-Instruct"),
    conf=0.4,
    mode="crop",
    names=names,
)

for out in viz.run("traffic.mp4", save="out.mp4"):
    pass
```

The first run downloads the weights. Expect the frame rate to drop hard whenever
a new object appears, because that is a forward pass in the middle of the loop.
Once a track has voted it costs nothing again.

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

`task="<OD>"` reports whatever Florence finds instead of only your classes.
Labels it returns that are not in `names` are dropped.

### End to end

```sh
pip install "vizor[hf]" ultralytics
```

Full mode, so Florence grounds the whole frame once and the refiner matches its
boxes to the tracks by IoU. That corrects the box and the confidence as well as
the class.

```python
import vizor as vz
from yolo import YOLO   # examples/yolo.py

names = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

viz = vz.Vizor(
    YOLO("yolo11n.pt"),
    vz.Florence("microsoft/Florence-2-base-ft", names=names),
    conf=0.5,
    mode="full",
    iou=0.5,     # how much a Florence box must overlap a track to count as it
    names=names,
)

for out in viz.run("traffic.mp4", save="out.mp4"):
    pass
```

Florence runs once per frame here, not once per object, because full mode has no
way to skip it. It reports no confidence either, so every box it matches comes
back at 1.0 and overwrites whatever the primary said.

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

### End to end

Nothing to install beyond the core, and nothing to download if you already have
two dumps and the video they came from.

```python
import vizor as vz

names = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

viz = vz.Vizor(
    vz.Pkl("yolov5nu_traffic3_preds.pkl", cols=vz.Pkl.ULTRALYTICS, names=names),
    vz.Pkl("Florence-2-base-ft_traffic3_preds.pkl", names=names),
    conf=0.5,
    mode="full",
    names=names,
)

frames = boxes = 0
for out in viz.run("traffic3.mp4", save="out.mp4"):
    frames += 1
    boxes += len(out)
print(f"{frames} frames, {boxes} boxes, {len(viz.refiner.cache)} tracks refined")
```

On the clip those dumps came from that prints one line and leaves an annotated
video behind.

```
3600 frames, 48756 boxes, 139 tracks refined
```

To make your own dump, pickle a list holding one array per frame, in the same
order as the video.

```python
import pickle

import vizor as vz
from yolo import YOLO

primary = YOLO("yolo11n.pt")
frames = [primary.track(f).data for f in vz.Video("traffic.mp4")]
with open("tracks.pkl", "wb") as f:
    pickle.dump(frames, f)
```

Those rows are already in vizor order, so replaying them needs no `cols`.

## A bigger detector as the secondary

Nothing about the refiner assumes the secondary is a VLM. It needs `find` or
`name`, and a heavier detector has both. That makes it the cheapest secondary to
reach for: it already speaks the primary's class list, so there is no menu to
write, no prompt to word and no key to keep. The pair is a small model at frame
rate with a big model's opinion on the boxes the small one was unsure about.

The `YOLO` adapter in `examples/yolo.py` fills either role, so the same class is
the primary and the secondary here, at two sizes.

```sh
pip install vizor ultralytics
```

`yolo11n` at 320 pixels runs on every frame. `yolo11m` at 960 sees only the crops
it was unsure of, once per track.

```python
import vizor as vz
from yolo import YOLO   # examples/yolo.py

viz = vz.Vizor(
    YOLO("yolo11n.pt", imgsz=320),    # fast, and wrong often enough to matter
    YOLO("yolo11m.pt", imgsz=960),    # slow, asked about one crop at a time
    conf=0.5,
    mode="crop",
)

for out in viz.run("traffic.mp4", save="out.mp4"):
    pass
print(f"{len(viz.refiner.cache)} tracks corrected")
```

`mode="full"` works too. The big model runs on the whole frame and its boxes are
matched to the tracks by IoU, so it fixes the box and the confidence as well as
the class. It costs a call on every frame that holds a doubtful track, rather
than one per track.

`python examples/bigger.py traffic.mp4` runs both models over a clip and counts
what the second one changed. On 300 frames of a 1280x720 traffic clip, on a
laptop CPU with no GPU, crop mode printed this.

```
300 frames, 2433 boxes, 4.5 fps
428 labels changed by yolo11m.pt
77 calls to the big model, 47 tracks in the vote cache
```

`--conf 0` never asks the big model, which is the small one on its own, and
`--mode full` is the third row. The box count is identical everywhere because
the refiner never adds or drops a box, only rewrites what is already there.

| | fps | boxes | labels changed | calls to yolo11m |
| --- | --- | --- | --- | --- |
| yolo11n at 320 alone | 26.6 | 2433 | 0 | 0 |
| the pair, crop mode | 4.5 | 2433 | 428 | 77 |
| the pair, full mode | 2.0 | 2433 | 157 | 274 |

Crop mode gets away with 77 calls across 300 frames because 47 of them are the
first sighting of a track and the rest are boxes the tracker never gave an id to.
Full mode cannot skip a frame that way, so it pays 274 and runs at half the
speed. It also changes fewer labels, because a track only takes the big model's
class when one of its boxes overlaps by `iou` or more, and an unmatched track is
left alone.

Whether the 428 changes are improvements is a question this does not answer. It
assumes the bigger model is right, which is the premise you are buying into when
you pick one as the secondary.

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
