# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Not part of the vizor package. AGPL because it uses the ultralytics adapter
# in yolo.py next to this file. See the header there.
"""YOLO on every frame, a hosted VLM on the boxes YOLO is unsure of.

Set your key first, the model never takes one from the command line:

    pip install ultralytics
    export GROQ_API_KEY=...
    python examples/live.py traffic.mp4 --save out.mp4

Use --api openai --model gpt-4o-mini to go through OpenAI instead.
"""

import argparse

from yolo import Yolo

from vizor import Vizor, Vlm


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", help="video file, camera index, or stream url")
    ap.add_argument("--weights", default="yolo11n.pt")
    ap.add_argument("--api", default="groq", choices=["groq", "openai"])
    ap.add_argument("--model", default="llama-3.2-11b-vision-preview")
    ap.add_argument("--conf", type=float, default=0.5, help="ask the VLM at or below this")
    ap.add_argument("--votes", type=int, default=1, help="VLM answers to collect per track")
    ap.add_argument("--save", default=None)
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    viz = Vizor(
        Yolo(args.weights),
        Vlm(args.model, api=args.api),
        conf=args.conf,
        mode="crop",
        votes=args.votes,
    )

    for out in viz.run(source, save=args.save, show=args.show):
        print(f"{len(out)} boxes, {len(viz.refiner.cache)} tracks refined", end="\r")


if __name__ == "__main__":
    main()
