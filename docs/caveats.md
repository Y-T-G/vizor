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

Crop mode batches, but only as far as one frame goes. `VLM` puts up to `chunk`
crops in a single request, and everything else falls back to one call per crop.
On the traffic clip that turns 569 crops into 528 requests at `votes=1`, a saving
of about 7 percent, because the vote cache has already removed most of the work.
The batches only get big when a lot of new objects appear at once. Raising
`votes` helps more, 2077 crops in 1632 requests at `votes=5`. Nothing batches
across frames, and `HF` does not batch at all yet.

The `conf` threshold is a single number applied to every class. A detector that is
badly calibrated on one class and well calibrated on another needs either a
per-class threshold or a lower global one that wastes calls elsewhere. Per-class
thresholds are not implemented.
