# How it works

Each frame goes through three steps.

1. The primary detects and tracks. You get boxes, confidences, classes and track ids.
2. Tracks at or below `conf` are handed to the secondary. Everything above it is left alone.
3. The secondary's answer is recorded as a vote against the track id. The running
   majority overwrites the class on this frame and on every later frame, whether
   or not the secondary runs again.

Step 3 is what makes this affordable. A car that stays in view for 300 frames
costs one VLM call, not 300.

The refiner never adds or drops a box. It only rewrites the class, and in full
mode the box and confidence too.

## The two modes

`mode="full"` runs the secondary on the whole frame, matches its boxes to the
tracks by IoU, and takes its box, confidence and class. Use it when the secondary
localises, so Florence-2, a heavier YOLO, or any open-vocabulary detector.

```python
viz = vz.Vizor(primary, vz.Florence(names=names), conf=0.5, mode="full", iou=0.5)
```

`iou` is the minimum overlap for a secondary box to count as the same object as a
track. Below it, the track is left alone.

`mode="crop"` cuts each low confidence track out of the frame and asks the
secondary what it is. The box stays as the primary drew it and only the class
changes. Use it when the secondary classifies but does not localise, which is
every chat VLM.

```python
viz = vz.Vizor(primary, vz.Vlm("gpt-4o-mini"), conf=0.5, mode="crop", votes=1)
```

`votes` is how many answers to collect per track before the secondary stops being
asked about it. At the default of 1, each track costs exactly one call for its
whole life. Raise it to trade calls for a majority that survives one bad answer.

The old names `"image"` and `"instance"` still work and mean `"full"` and
`"crop"`.

## The vote cache

[`Vote`][vizor.vote.Vote] is an LRU keyed on track id. Each id holds a bounded
deque of class ids, and the winner is the most common entry.

```python
from vizor import Vote

v = Vote(size=256, hist=25)
v.add(7, 2)
v.add(7, 2)
v.add(7, 5)
v.get(7)   # 2
```

`size` caps how many track ids are remembered at once, and the least recently
used id is dropped when it is exceeded. `hist` caps how many votes each id keeps,
so an object that changes appearance can eventually change label.

Ids below zero are ignored. Untracked boxes all carry `id = -1`, and pooling them
into one cache entry would make every untracked object share a label.

## Numbers

The measurements below come from YOLOv5n tracks and Florence-2 grounding output
for a 3600 frame traffic clip, replayed from disk with
[`Pkl`][vizor.models.pkl.Pkl]. Reproduce them with `python examples/bench.py`.

At `conf=0.5` there are 8457 boxes below the threshold across the video, and the
vote cache turns them into 569 secondary calls.

```
 conf   boxes  relabelled  ms/frame  low conf  vlm calls
 0.30   48756        2518      0.11      2331        336
 0.50   48756        2342      0.15      8457        569
 0.90   48756        6443      0.22     47579        596
```

`relabelled` and `ms/frame` come from full mode against the Florence output.
`low conf` and `vlm calls` come from crop mode with a stub secondary. The
`ms/frame` column is the refiner's own cost, which is the IoU match and the vote
lookup, and it moves by a few hundredths between runs. The secondary's inference
cost is on top of that and is the only part that matters in practice.

The table counts how many labels changed. It does not say how many of the changes
were right, because there is no ground truth here. See [caveats](caveats.md).
