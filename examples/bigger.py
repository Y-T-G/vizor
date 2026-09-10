# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Not part of the vizor package. AGPL because it uses the ultralytics adapter
# in yolo.py next to this file. See the header there.
"""A small YOLO corrected by a bigger one. No VLM, no API key, no network.

The secondary does not have to be a VLM. Anything with a ``name`` or a ``find``
method will do, and a heavier detector is the cheapest thing to reach for: it
speaks the same class list as the primary, so there is no prompt to write.

    pip install vizor ultralytics
    python examples/bigger.py traffic.mp4

It runs the small model alone and then the pair, and reports how many labels
the big model changed and what it cost.
"""

import argparse
import time

from yolo import YOLO

from vizor import Refiner, Video


class Counting(YOLO):
    """The same adapter, keeping a tally of how often it was asked."""

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.calls = 0

    def find(self, img, names=None):
        self.calls += 1
        return super().find(img, names)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source")
    ap.add_argument("--small", default="yolo11n.pt")
    ap.add_argument("--big", default="yolo11m.pt")
    ap.add_argument("--imgsz", type=int, default=320, help="input size for the small model")
    ap.add_argument("--big-imgsz", type=int, default=960, help="input size for the big model")
    ap.add_argument("--mode", default="crop", choices=["crop", "full"])
    ap.add_argument("--conf", type=float, default=0.5, help="ask the big model at or below this")
    ap.add_argument("--frames", type=int, default=0, help="stop after this many, 0 for all")
    ap.add_argument("--save", default=None)
    args = ap.parse_args()

    small = YOLO(args.small, imgsz=args.imgsz)
    big = Counting(args.big, imgsz=args.big_imgsz)
    # crop mode asks the big model about one box at a time and caches the answer
    # against the track id. full mode runs it on the whole frame instead, which
    # fixes the box as well as the class but costs a call every frame.
    refiner = Refiner(big, conf=args.conf, mode=args.mode)

    frames = boxes = changed = 0
    start = time.perf_counter()
    video = Video(args.source)
    writer = None
    try:
        for frame in video:
            tracks = small.track(frame)
            before = tracks.cls            # a copy, so the refiner cannot touch it
            out = refiner.run(tracks, img=frame)
            frames += 1
            boxes += len(out)
            changed += int((out.cls != before).sum())
            if args.save:
                from vizor import Writer

                writer = writer or Writer(args.save, fps=video.fps)
                writer.write(out.draw(frame))
            if args.frames and frames >= args.frames:
                break
    finally:
        video.close()
        if writer:
            writer.close()

    took = time.perf_counter() - start
    print(f"{frames} frames, {boxes} boxes, {frames / took:.1f} fps")
    print(f"{changed} labels changed by {args.big}")
    print(f"{big.calls} calls to the big model, {len(refiner.cache)} tracks in the vote cache")


if __name__ == "__main__":
    main()
