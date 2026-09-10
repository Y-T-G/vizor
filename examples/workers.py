"""What `workers` costs and what it buys, with no model and no network.

The secondary here is a stub that sleeps instead of calling an API, so the
numbers come out the same on any machine:

    python examples/workers.py
"""

import argparse
import time

import numpy as np

from vizor import Refiner, Tracks
from vizor.models.base import Model

NAMES = {0: "person", 7: "truck"}
FRAME = np.zeros((200, 200, 3), np.uint8)


class Slow(Model):
    """Stands in for a hosted VLM, answering after a fixed delay."""

    def __init__(self, delay):
        self.delay = delay
        self.calls = 0

    def batch(self, crops, names=None, hints=None):
        self.calls += 1
        time.sleep(self.delay)
        return [7] * len(crops)


def doubtful():
    """One tracked box the primary is unsure of, class 0, id 1."""
    return Tracks(np.array([[0, 0, 40, 40, 0.2, 0, 1]], np.float32), names=NAMES)


def measure(workers, frames, delay, budget):
    model = Slow(delay)
    r = Refiner(model, conf=0.5, mode="crop", names=NAMES, workers=workers)
    seen, start = [], time.perf_counter()
    for _ in range(frames):
        t0 = time.perf_counter()
        seen.append(int(r.run(doubtful(), img=FRAME).cls[0]))
        # hold the primary's frame rate, so the only variable is the secondary
        time.sleep(max(0.0, budget - (time.perf_counter() - t0)))
    took = time.perf_counter() - start
    r.close()
    # the first frame that carries the corrected class, 7
    fixed = next((i for i, c in enumerate(seen) if c == 7), None)
    return frames / took, fixed, model.calls, seen


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--delay", type=float, default=0.4, help="secondary round trip, seconds")
    ap.add_argument("--fps", type=float, default=30.0, help="what the primary alone would do")
    ap.add_argument("--workers", type=int, nargs="+", default=[0, 2])
    args = ap.parse_args()

    print(f"{args.frames} frames, a {args.delay * 1000:.0f} ms secondary, "
          f"a {args.fps:.0f} fps primary\n")
    print(f"{'workers':>8} {'fps':>6} {'corrected from':>15} {'requests':>9}  labels")
    for w in args.workers:
        fps, fixed, calls, seen = measure(w, args.frames, args.delay, 1.0 / args.fps)
        mark = f"frame {fixed}" if fixed is not None else "never"
        print(f"{w:>8} {fps:>6.0f} {mark:>15} {calls:>9}  {''.join(str(c) for c in seen)}")


if __name__ == "__main__":
    main()
