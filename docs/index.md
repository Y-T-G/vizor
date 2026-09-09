# vizor

A small detector is fast enough to run on every frame but gets classes wrong. A
vision-language model gets them right but is far too slow to run on every frame,
and often gives you no boxes at all.

Vizor runs both and keeps the useful half of each. Boxes and track ids come from
the detector. Class labels come from the VLM. Every VLM answer is cached against
the track id, so the same object is never asked about twice.

The code is at [github.com/Y-T-G/vizor](https://github.com/Y-T-G/vizor). The
`examples/` directory holds runnable versions of everything on this page.

## Install

Install the core, then the extra for whichever model you plan to use.

```sh
pip install vizor              # core: numpy and OpenCV
pip install "vizor[api]"       # hosted VLMs: OpenAI, Gemini, or any compatible url
pip install "vizor[hf]"        # local Florence-2 or another transformers VLM
```

The full list of extras is on the [install page](install.md). Skipping the extra
gives you an `ImportError` on the first frame, not at import time, because model
wrappers are only imported when you construct one.

## Quick start

This runs a detector on every frame and sends the boxes it is unsure of to a
hosted VLM, one crop per box. `YOLO` comes from `examples/yolo.py`, which is not
part of the installed package.

```python
import vizor as vz
from yolo import YOLO

# YOLO detects and tracks every frame. The VLM only sees boxes YOLO scored at or
# below conf, cropped out one at a time, and its answer is cached against the
# track id, so each object costs one call no matter how long it stays in view.
viz = vz.Vizor(YOLO("yolo11n.pt"), vz.VLM("gpt-4o-mini"), conf=0.5, mode="crop")

# run() yields the refined boxes for each frame and writes the annotated video
# as it goes. Drop save= and nothing is written.
for out in viz.run("traffic.mp4", save="out.mp4"):
    print(len(out), "boxes")
```

That writes an annotated `out.mp4` and yields the refined boxes for every frame.
The API key is read from `OPENAI_API_KEY`. Never pass it as a literal in code you
commit.

Florence-2 works instead, running locally and grounding the whole frame in one
pass.

```python
import vizor as vz
from yolo import YOLO

# the class list is Florence's prompt, so it looks for these three and nothing else
names = {0: "person", 2: "car", 7: "truck"}

# full mode grounds the whole frame in one pass, then matches Florence's boxes to
# the tracks by IoU. A match overwrites the box and confidence, not just the class.
viz = vz.Vizor(YOLO("yolo11n.pt"), vz.Florence(names=names), conf=0.5, mode="full")

# 0 is the first webcam. show=True opens a window, q or Esc closes it.
for out in viz.run(0, show=True):
    pass
```

Florence returns boxes as well as labels, so the refiner corrects the box too,
not just the class. That difference is what the two modes are for, and it is
covered in [how it works](how-it-works.md).

## What you get back

Every frame yields a [`Tracks`][vizor.boxes.Tracks], which wraps a float32 array
of shape `(N, 7)` holding `[x1, y1, x2, y2, conf, cls, id]`. Untracked boxes
carry `id = -1`.

```python
out.data              # the raw (N, 7) array
out.boxes             # (N, 4) xyxy, a view, so writing to it edits data
out.conf              # (N,) confidences, also a view
out.cls, out.ids      # (N,) ints, copies, so writing to them changes nothing
out[0]                # one Track
out[out.conf > 0.8]   # a new Tracks holding a copy of the matching rows
out.draw()            # draw the boxes on the frame it came from, return the image
```

Indexing with an int gives one [`Track`][vizor.boxes.Track]. Indexing with a mask
or a slice gives a new `Tracks`, and because numpy copies on fancy indexing,
writing to that copy does not touch the original.
