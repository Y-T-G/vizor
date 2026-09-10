# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Not part of the vizor package. AGPL because it uses the ultralytics adapter
# in yolo.py next to this file. See the header there.
"""Ask a VLM an attribute question about each tracked person.

One frame is often not enough to tell. Collage mode gathers several crops of
the same track, spaced out over time, tiles them into one image and asks about
that once.

    pip install "vizor[api]" ultralytics
    export GEMINI_API_KEY=...
    python examples/collage.py street.mp4 --save out.mp4

The classes are the answers you will accept, so --classes is the question. The
detector only ever says "person", and the VLM replaces that with one of them.
"""

import argparse

from yolo import YOLO

from vizor import VLM, Vizor

# ultralytics class 0 is person, so the primary is told to report nothing else
PERSON = 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", help="video file, camera index, or stream url")
    ap.add_argument("--classes", default="man,woman", help="what the VLM may answer")
    ap.add_argument("--weights", default="yolo11n.pt")
    ap.add_argument("--api", default="gemini", choices=["gemini", "openai"])
    ap.add_argument("--model", default="gemini-3.1-flash-lite")
    ap.add_argument("--samples", type=int, default=6, help="crops per collage")
    ap.add_argument("--every", type=int, default=8, help="frames between one track's crops")
    ap.add_argument("--cell", type=int, nargs=2, default=(96, 192),
                    help="cell width and height, tall by default because people are")
    ap.add_argument("--cols", type=int, default=3)
    ap.add_argument("--workers", type=int, default=0,
                    help="background threads for the VLM, 0 waits for every answer")
    ap.add_argument("--save", default=None)
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    names = {i: n.strip() for i, n in enumerate(args.classes.split(",")) if n.strip()}
    source = int(args.source) if args.source.isdigit() else args.source
    viz = Vizor(
        YOLO(args.weights, classes=[PERSON]),   # people only, so nothing else gets asked about
        VLM(args.model, api=args.api),
        conf=1.0,                # the question is not about doubt, so every track is asked
        mode="collage",
        names=names,             # the menu the VLM picks from, not the detector's classes
        samples=args.samples,
        every=args.every,
        cell=tuple(args.cell),
        cols=args.cols,
        workers=args.workers,
    )

    # a track wears the detector's class until its collage is full and answered,
    # which takes samples x every frames at the earliest
    with viz:
        for out in viz.run(source, save=args.save, show=args.show):
            print(f"{len(out)} people, {len(viz.refiner.cache)} answered, "
                  f"{len(viz.refiner.shots)} part way", end="\r")


if __name__ == "__main__":
    main()
