# Examples

Each script is runnable as it stands. Run them from the repo root, not from this
directory, because two of them look for data under `nbs/`.

| Script | What it does | Needs |
| --- | --- | --- |
| [`workers.py`](workers.py) | measures blocking against background refinement | nothing |
| [`replay.py`](replay.py) | refines saved predictions and writes an annotated video | saved predictions |
| [`bench.py`](bench.py) | counts labels changed and secondary calls saved | saved predictions |
| [`bigger.py`](bigger.py) | a small YOLO corrected by a big one, and what that costs | ultralytics |
| [`live.py`](live.py) | YOLO on a video or webcam, a hosted VLM on the doubtful boxes | ultralytics, an API key |
| [`collage.py`](collage.py) | asks a VLM one question per person, from crops over time | ultralytics, an API key |
| [`yolo.py`](yolo.py) | the ultralytics adapter the other scripts import | ultralytics |

## The one that needs nothing

`workers.py` stands a sleeping stub in for the secondary, so it needs no model,
no key and no video.

```sh
python examples/workers.py
```

It prints what the `workers` setting trades away.

```
20 frames, a 400 ms secondary, a 30 fps primary

 workers    fps  corrected from  requests  labels
       0     19         frame 0         1  77777777777777777777
       2     30        frame 13         1  00000000000007777777
```

## The two that need saved predictions

`replay.py` and `bench.py` read a YOLOv5n track dump and a Florence-2 grounding
dump for a traffic clip, and `replay.py` also wants the clip itself.

```
nbs/yolov5nu_traffic3_preds.pkl
nbs/Florence-2-base-ft_traffic3_preds.pkl
nbs/traffic3.mp4
```

Those files are not in the repo, so both scripts fail with `FileNotFoundError` on
a fresh clone. Point them somewhere else with `--primary`, `--secondary` and
`--video`, or make your own dump: pickle a list holding one array of boxes per
frame, then load it with [`Pkl`](https://y-t-g.github.io/vizor/models/#pkl).

```sh
python examples/replay.py --save out.mp4
python examples/bench.py
```

`replay.py` writes an annotated video, then reports the frame rate and how many
tracks ended up in the vote cache.

```
3600 frames in 10.2s (353 fps)
139 tracks in the vote cache
```

`bench.py` prints how many labels changed at each `conf`, and how many secondary
calls the vote cache saved.

## The one that needs no key

`bigger.py` uses a heavier YOLO as the secondary instead of a VLM, so it needs
ultralytics and a video and nothing else. It runs both models over the clip and
reports what the big one changed.

```sh
pip install vizor ultralytics
python examples/bigger.py traffic.mp4 --frames 300
```

This is what it printed on 300 frames of a 1280x720 traffic clip, on a laptop
CPU with no GPU. Your frame rate will differ, the shape of the answer will not.

```
300 frames, 2433 boxes, 4.5 fps
428 labels changed by yolo11m.pt
77 calls to the big model, 47 tracks in the vote cache
```

`--conf 0` never asks the big model, which gives you the small one's own numbers
to compare against. `--mode full` runs it on the whole frame instead of the
crops, which also corrects the boxes.

## The two that cost money

`live.py` runs a real detector on a real video and sends the doubtful crops to a
hosted VLM. Install ultralytics and set a key first, or it will not start.

```sh
pip install "vizor[api]" ultralytics
export GEMINI_API_KEY=...
python examples/live.py traffic.mp4 --save out.mp4
```

Add `--workers 2` to stop the frame loop waiting on the VLM, and `--show` for a
window. `--api openai --model gpt-4o-mini` goes through OpenAI instead, with
`OPENAI_API_KEY` set.

`collage.py` asks a different kind of question. It tracks people, gathers six
crops of each one spread over 40 frames, tiles them into a single image and asks
the VLM which of your classes that person is. The detector never knew the answer,
so this is not correcting it, it is adding a label the detector could not give.

```sh
python examples/collage.py street.mp4 --classes man,woman --save out.mp4
```

`--classes` is the question. `--samples` and `--every` set how many crops go in a
collage and how far apart they are taken, so together they decide how long a
track waits before it is answered.

## Licensing

`yolo.py`, `bigger.py`, `live.py` and `collage.py` are `AGPL-3.0-or-later`,
because they use ultralytics. The rest of this repository is Apache-2.0. Read the
header in `yolo.py` before copying it into your own project.
