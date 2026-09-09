# Pipeline

`Vizor` is the front door. It owns a primary model and a `Refiner`, and drives
them over a video. `Refiner` is the part that decides what to send to the
secondary and what to do with the answer, and you can use it on its own if you
already have detections.

::: vizor.core.Vizor

::: vizor.refine.Refiner

::: vizor.vote.Vote
