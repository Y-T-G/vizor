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
pip install "vizor[yolo]"      # ultralytics primary
pip install "vizor[hf]"        # local Florence-2 or another transformers VLM
pip install "vizor[api,groq]"  # hosted VLMs over OpenAI or Groq
```

Model wrappers are imported on first use, so `import vizor` never pulls in torch
if you are not using a torch model.

## Quick start

YOLO on every frame, a hosted VLM on the boxes YOLO is unsure of:

```python
import vizor as vz

viz = vz.Vizor(vz.Yolo("yolo11n.pt"), vz.Vlm("gpt-4o-mini"), conf=0.5, mode="crop")

for out in viz.run("traffic.mp4", save="out.mp4"):
    print(len(out), "boxes")
```

That writes an annotated `out.mp4` and yields the refined boxes for every frame.
The key is read from `OPENAI_API_KEY`, never passed in code.

Florence-2 instead, running locally and grounding the whole frame in one pass:

```python
import vizor as vz

names = {0: "person", 2: "car", 7: "truck"}
viz = vz.Vizor(vz.Yolo("yolo11n.pt"), vz.Florence(names=names), conf=0.5, mode="full")

for out in viz.run(0, show=True):   # 0 is the first webcam
    pass
```

Florence gives boxes as well as labels, so the refiner can correct the box too,
not just the class.

To try it with no models at all, replay the cached predictions:

```sh
python examples/replay.py --save out.mp4
```

On this machine that prints `3600 frames in 9.6s (376 fps)` and leaves 139 tracks
in the vote cache.

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

## Numbers

The repo ships YOLOv5n tracks and Florence-2 grounding output for `traffic3.mp4`,
3600 frames. Reproduce the table with `python examples/bench.py`:

```
 conf   boxes  relabelled  ms/frame  low conf  vlm calls
 0.30   48756        2518      0.11      2331        336
 0.50   48756        2342      0.15      8457        569
 0.90   48756        6443      0.22     47579        596
```

`relabelled` and `ms/frame` come from full mode against the Florence output.
`low conf` and `vlm calls` come from crop mode with a stub secondary. At
`conf=0.5` there are 8457 boxes below the threshold across the video, and the
vote cache turns them into 569 calls. The `ms/frame` column is the refiner's own
cost, which is the IoU match and the vote lookup. The secondary's inference cost
is on top of that and is the only part that matters in practice.

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

Bundled models: `Yolo` (primary or secondary), `Vlm` (OpenAI, Groq, or any
OpenAI-compatible url), `Hf` (a local transformers chat VLM), `Florence`
(Florence-2 as an open-vocabulary detector), `Pkl` (replay saved predictions).

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

There is no evaluation against ground truth. The table above counts how many
labels changed, not how many of the changes were right.

## Links

- [Ultralytics](https://docs.ultralytics.com/) for the primary detector and trackers
- [Florence-2](https://huggingface.co/microsoft/Florence-2-base-ft) for open-vocabulary grounding
- [Groq](https://console.groq.com/docs) and [OpenAI](https://platform.openai.com/docs/guides/vision) for hosted vision models
