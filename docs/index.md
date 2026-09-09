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
pip install "vizor[api]"       # hosted VLMs over an OpenAI-compatible API
pip install "vizor[hf]"        # local Florence-2 or another transformers VLM
```

The full list of extras is on the [install page](install.md). Skipping the extra
gives you an `ImportError` on the first frame, not at import time, because model
wrappers are only imported when you construct one.

## Quick start

This runs YOLO on every frame and sends the boxes YOLO is unsure of to a hosted
VLM, one crop per box. `Yolo` is the adapter from `examples/yolo.py`, which is
not part of the installed package. [Licensing](models.md#licensing) explains why.

```python
import vizor as vz
from yolo import Yolo

viz = vz.Vizor(Yolo("yolo11n.pt"), vz.Vlm("gpt-4o-mini"), conf=0.5, mode="crop")

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
from yolo import Yolo

names = {0: "person", 2: "car", 7: "truck"}
viz = vz.Vizor(Yolo("yolo11n.pt"), vz.Florence(names=names), conf=0.5, mode="full")

for out in viz.run(0, show=True):   # 0 is the first webcam
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
out.boxes             # (N, 4) xyxy
out.conf, out.cls, out.ids
out[0]                # a single Track
out[out.conf > 0.8]   # a copy holding the rows that match
out.draw()            # annotate the frame it came from and return it
```

Indexing with an int gives one [`Track`][vizor.boxes.Track]. Indexing with a mask
or a slice gives a new `Tracks`, and because numpy copies on fancy indexing,
writing to that copy does not touch the original.
