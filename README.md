# vizor

A small detector is fast enough to run on every frame but gets classes wrong. A
vision-language model gets them right but is far too slow to run on every frame,
and often gives you no boxes at all. Vizor runs both and keeps the useful half of
each: boxes and track ids come from the detector, class labels come from the VLM,
and every VLM answer is cached against the track id so the same object is never
asked about twice.

Full documentation is at [y-t-g.github.io/vizor](https://y-t-g.github.io/vizor/).
Everything below runs from this repo, and `examples/` holds each snippet as a
script you can run.

## Install

The core needs numpy and OpenCV only. The model wrappers are optional extras, so
install the one you actually use:

```sh
pip install vizor              # core
pip install "vizor[hf]"        # local Florence-2 or another transformers VLM
pip install "vizor[api,groq]"  # hosted VLMs over OpenAI or Groq
```

Model wrappers are imported on first use, so `import vizor` never pulls in torch
if you are not using a torch model.

vizor ships no detector. It ships the base class and one working adapter you
copy. `examples/yolo.py` is about 60 lines and wraps ultralytics, which is
AGPL-3.0, so it lives outside the package and carries its own licence header.
Read [detectors and licensing](#detectors-and-licensing) before you use it.

## Quick start

A detector on every frame, a hosted VLM on the boxes it is unsure of. `YOLO`
comes from `examples/yolo.py`, not from `vizor`:

```python
import vizor as vz
from yolo import YOLO

viz = vz.Vizor(YOLO("yolo11n.pt"), vz.Vlm("gpt-4o-mini"), conf=0.5, mode="crop")

for out in viz.run("traffic.mp4", save="out.mp4"):
    print(len(out), "boxes")
```

That writes an annotated `out.mp4` and yields the refined boxes for every frame.
The key is read from `OPENAI_API_KEY`, never passed in code.

Florence-2 instead, running locally and grounding the whole frame in one pass:

```python
import vizor as vz
from yolo import YOLO

names = {0: "person", 2: "car", 7: "truck"}
viz = vz.Vizor(YOLO("yolo11n.pt"), vz.Florence(names=names), conf=0.5, mode="full")

for out in viz.run(0, show=True):   # 0 is the first webcam
    pass
```

Florence gives boxes as well as labels, so the refiner can correct the box too,
not just the class.

To try it with no models at all, replay predictions you saved earlier:

```sh
python examples/replay.py --save out.mp4
```

That runs the refiner over the saved frames and writes an annotated video, so you
can see what the refinement does without loading a model or spending a call.

## How it works

Each frame goes through three steps.

1. The primary detects and tracks. You get boxes, confidences, classes and track ids.
2. Tracks at or below `conf` are handed to the secondary. Everything above it is
   left alone.
3. The secondary's answer is recorded as a vote against the track id. The running
   majority overwrites the class on this frame and on every later frame, whether
   or not the secondary runs again.

Step 3 is what makes this affordable. A car that stays in view for 300 frames
costs one VLM call, not 300.

There are two modes.

`mode="full"` runs the secondary on the whole frame, matches its boxes to the
tracks by IoU, and takes its box, confidence and class. Use it when the secondary
localises: Florence-2, a heavier YOLO, any open-vocabulary detector.

`mode="crop"` cuts each low confidence track out of the frame and asks the
secondary what it is. The box stays as the primary drew it and only the class
changes. Use it when the secondary classifies but does not localise, which is
every chat VLM.

The refiner never adds or drops a box. It only rewrites the class, and in full
mode the box and confidence too.

## API

```python
vz.Vizor(primary, secondary=None, conf=0.5, mode="full", names=None, **kw)
```

- `conf` sends tracks at or below this confidence to the secondary.
- `mode` is `"full"` or `"crop"`.
- `names` maps class ids to strings. Defaults to whatever the primary reports.
- `iou` is the minimum overlap to match a secondary box to a track, full mode only.
- `votes` is how many answers to collect per track before the secondary stops
  being asked, crop mode only. Default 1.
- `size` and `hist` cap the vote cache at that many track ids and that many votes
  each.

`viz.run(src, save=None, show=False)` yields refined `Tracks` for every frame of a
file, a camera index, or a stream url. `viz.step(img)` does one frame.
`viz.save(src, out)` runs the whole thing and writes the annotated video.
`viz.reset()` clears the votes and the primary's tracker between videos.

`Tracks` wraps a float32 array of shape `(N, 7)` holding
`[x1, y1, x2, y2, conf, cls, id]`, with `id = -1` for untracked boxes:

```python
out.data      # the raw array
out.boxes     # (N, 4) xyxy
out.conf, out.cls, out.ids
out[0]        # a single Track
out[out.conf > 0.8]   # a copy holding the rows that match
out.draw()    # annotate the frame it came from and return it
```

Indexing with an int gives you one `Track`. Indexing with a mask or a slice gives
you a new `Tracks`, and because numpy copies on fancy indexing, writing to it does
not touch the original.

Bundled models: `Vlm` (OpenAI, Groq, or any OpenAI-compatible url), `Hf` (a local
transformers chat VLM), `Florence` (Florence-2 as an open-vocabulary detector),
`Pkl` (replay saved predictions). All of them are secondaries. The primary is
yours to bring.

## Detectors and licensing

vizor is Apache-2.0. Ultralytics is AGPL-3.0, and AGPL says a work that combines
with it must also be AGPL-3.0. A Python module that imports ultralytics forms
that combined work when it runs, so shipping a YOLO wrapper inside an Apache-2.0
wheel would put the two licences in conflict.

So the wrapper is not in the package. It is `examples/yolo.py`, marked
`AGPL-3.0-or-later`, and pip never installs it. Nothing that pip installs imports
ultralytics.

What that means for you. If your own project is AGPL-3.0, or you hold an
Ultralytics Enterprise licence, copy `examples/yolo.py` and use it. If your
project is closed source or permissively licensed, write an adapter for a
detector whose licence you can live with. The interface is one method, and
[writing your own model](#writing-your-own-model) below shows it.

I am not a lawyer and this is not legal advice. If the answer matters
commercially, ask one.

## Writing your own model

Subclass `vizor.Model` and implement the method your role needs. A primary
implements `track`, a full-mode secondary implements `find`, a crop-mode
secondary implements `name`. Anything you leave out raises on the first frame
with a message saying which role is missing, rather than silently doing nothing.

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
your wrapper if the model wants RGB. `name` returns a class id, or `None` if the
model is not sure, and `None` records no vote.

## Caveats

The vote cache is keyed on the track id, so it is only as good as the tracker. If
the tracker swaps ids between two nearby objects, the refined class follows the id
and lands on the wrong object. Raising `hist` makes a single bad frame matter less
but does not fix an id swap.

Untracked boxes all carry `id = -1`, so the cache skips them. They are refined on
the frame they appear on and never remembered.

Florence-2 reports no confidence, so every box it returns comes back at 1.0. In
full mode that overwrites the primary's confidence with a number that means
nothing. Set `best=False` if you would rather every matching box vote instead of
only the highest-IoU one, but the confidence problem stays.

Nothing here is batched. Crop mode sends one request per low confidence track, one
at a time. Batching the crops of a frame into a single request would cut the
latency a lot, and is the obvious thing to add next.

## Links

- [Ultralytics](https://docs.ultralytics.com/) for the detector and trackers used in `examples/yolo.py`, [AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE)
- [Florence-2](https://huggingface.co/microsoft/Florence-2-base-ft) for open-vocabulary grounding
- [Groq](https://console.groq.com/docs) and [OpenAI](https://platform.openai.com/docs/guides/vision) for hosted vision models
