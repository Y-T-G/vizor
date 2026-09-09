"""Count what the refiner changes, and how many secondary calls it saves.

Everything runs off the cached predictions in nbs/, so the numbers come out the
same on any machine:

    python examples/bench.py
"""

import argparse
import time

import numpy as np

from vizor import Vizor
from vizor.models.base import Model
from vizor.models.pkl import Pkl

NAMES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
PRIMARY = "nbs/yolov5nu_traffic3_preds.pkl"
SECONDARY = "nbs/Florence-2-base-ft_traffic3_preds.pkl"


class Counter(Model):
    """Stands in for a real VLM and counts how often it would have been called."""

    def __init__(self):
        self.calls = 0

    def name(self, crop, names=None, hint=None):
        self.calls += 1
        return 2


def full(conf):
    primary = Pkl(PRIMARY, cols=Pkl.ULTRALYTICS, names=NAMES)
    viz = Vizor(primary, Pkl(SECONDARY, names=NAMES), names=NAMES, conf=conf, mode="full")
    boxes = changed = 0
    took = 0.0
    for _ in range(len(primary)):
        before = np.asarray(primary.frames[primary.i], np.float32)[:, 6].astype(int)
        start = time.perf_counter()
        out = viz.step(None)
        took += time.perf_counter() - start
        boxes += len(out)
        changed += int((out.cls != before).sum())
    return boxes, changed, took / len(primary)


def crop(conf):
    primary = Pkl(PRIMARY, cols=Pkl.ULTRALYTICS, names=NAMES)
    counter = Counter()
    viz = Vizor(primary, counter, names=NAMES, conf=conf, mode="crop", votes=1, size=512)
    frame = np.zeros((720, 1280, 3), np.uint8)
    low = 0
    for _ in range(len(primary)):
        low += int((viz.step(frame).conf <= conf).sum())
    return low, counter.calls


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--conf", type=float, nargs="+", default=[0.3, 0.5, 0.9])
    args = ap.parse_args()

    print(f"{'conf':>5} {'boxes':>7} {'relabelled':>11} {'ms/frame':>9} "
          f"{'low conf':>9} {'vlm calls':>10}")
    for conf in args.conf:
        boxes, changed, per_frame = full(conf)
        low, calls = crop(conf)
        print(f"{conf:>5.2f} {boxes:>7} {changed:>11} {1000 * per_frame:>9.2f} "
              f"{low:>9} {calls:>10}")


if __name__ == "__main__":
    main()
