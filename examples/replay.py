"""Refine cached YOLOv5n tracks with cached Florence-2 grounding output.

Runs offline on the pickles in nbs/, so it needs no GPU and no network.

    python examples/replay.py --save out.mp4
"""

import argparse
import time

from vizor import Vizor
from vizor.models.pkl import Pkl

NAMES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", default="nbs/traffic3.mp4")
    ap.add_argument("--primary", default="nbs/yolov5nu_traffic3_preds.pkl")
    ap.add_argument("--secondary", default="nbs/Florence-2-base-ft_traffic3_preds.pkl")
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--save", default=None, help="write an annotated video here")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    viz = Vizor(
        # the YOLO dump is in ultralytics column order, so reindex it
        Pkl(args.primary, cols=Pkl.ULTRALYTICS, names=NAMES),
        # the Florence dump is already in vizor order
        Pkl(args.secondary, names=NAMES),
        conf=args.conf,
        mode="full",
        names=NAMES,
    )

    # each frame pulls the next row out of both pickles, matches them by IoU, and
    # votes the result against the track id. The video is only there to draw on.
    frames = 0
    start = time.perf_counter()
    for _ in viz.run(args.video, save=args.save, show=args.show):
        frames += 1
    took = time.perf_counter() - start
    print(f"{frames} frames in {took:.1f}s ({frames / took:.0f} fps)")
    print(f"{len(viz.refiner.cache)} tracks in the vote cache")


if __name__ == "__main__":
    main()
