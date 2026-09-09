# Caveats

The vote cache is keyed on the track id, so it is only as good as the tracker. If
the tracker swaps ids between two nearby objects, the refined class follows the id
and lands on the wrong object. Raising `hist` makes a single bad frame matter
less, but it does not fix an id swap.

Untracked boxes all carry `id = -1`, so the cache skips them. They are refined on
the frame they appear on and never remembered. If your primary does not track,
every frame pays full price.

Florence-2 reports no confidence, so every box it returns comes back at 1.0. In
full mode that overwrites the primary's confidence with a number that means
nothing. Set `best=False` if you would rather every matching box vote instead of
only the highest-IoU one, but the confidence problem stays either way.

Nothing here is batched. Crop mode sends one request per low confidence track, one
at a time. Batching a frame's crops into a single request would cut the latency a
lot, and it is the obvious thing to add next.

There is no evaluation against ground truth. The table in
[how it works](how-it-works.md) counts how many labels changed, not how many of
the changes were right. Someone running this against a labelled set would be the
fastest way to find out whether the whole idea holds up, and I have not done it.

The `conf` threshold is a single number applied to every class. A detector that is
badly calibrated on one class and well calibrated on another needs either a
per-class threshold or a lower global one that wastes calls elsewhere. Per-class
thresholds are not implemented.
