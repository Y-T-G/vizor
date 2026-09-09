"""The pipeline: a primary detector every frame, a secondary model when needed."""

from .boxes import Tracks
from .refine import Refiner
from .utils.video import Video, Writer

__all__ = ["Vizor"]


class Vizor:
    """Run a fast tracker on every frame and refine it with a slower model.

    The primary gives boxes and track ids at frame rate. The secondary is only
    asked about tracks the primary is unsure of, and its answers are cached
    against the track id, so the cost is paid once per object rather than once
    per frame.

    Args:
        primary: model with a ``track(img)`` method returning :class:`~vizor.boxes.Tracks`.
        secondary: model with ``find`` (full mode) or ``name`` (crop mode). Optional.
        conf: tracks at or below this confidence go to the secondary.
        mode: ``"full"`` runs the secondary on the whole frame, ``"crop"`` on each box.
        names: class id to name mapping. Defaults to whatever the primary reports.

    Remaining keyword arguments go to :class:`~vizor.refine.Refiner`.
    """

    def __init__(self, primary, secondary=None, conf=0.5, mode="full", names=None, **kw):
        self.primary = primary
        self.refiner = Refiner(secondary, conf=conf, mode=mode, names=names, **kw)

    @property
    def secondary(self):
        return self.refiner.model

    @property
    def names(self):
        return self.refiner.names or getattr(self.primary, "names", None)

    def reset(self):
        """Clear the vote cache and any state the primary keeps between frames."""
        self.refiner.reset()
        reset = getattr(self.primary, "reset", None)
        if callable(reset):
            reset()

    def step(self, img, preds=None):
        """Process one frame and return the refined :class:`~vizor.boxes.Tracks`."""
        tracks = self.primary.track(img)
        if not isinstance(tracks, Tracks):
            tracks = Tracks(tracks)
        if tracks.names is None:
            tracks.names = getattr(self.primary, "names", None)
        return self.refiner.run(tracks, img=img, preds=preds)

    def run(self, src, save=None, show=False, fourcc="mp4v"):
        """Yield refined tracks for every frame of ``src``.

        ``src`` is anything :class:`~vizor.utils.video.Video` opens: a file path,
        a camera index, or a stream url. Pass ``save`` to also write an annotated
        video, and ``show`` to display it in a window.

        Each yielded ``Tracks`` carries its frame, so ``out.draw()`` needs no
        argument. Drawing happens on the frame itself, so copy it first if you
        want the original.
        """
        video = Video(src)
        writer = Writer(save, fps=video.fps, fourcc=fourcc) if save else None
        try:
            for frame in video:
                out = self.step(frame)
                yield out
                if writer or show:
                    drawn = out.draw(frame)
                    if writer:
                        writer.write(drawn)
                    if show:
                        import cv2

                        cv2.imshow("vizor", drawn)
                        if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                            break
        finally:
            video.close()
            if writer:
                writer.close()
            if show:
                import cv2

                try:
                    cv2.destroyWindow("vizor")
                except cv2.error:
                    pass

    def save(self, src, out, show=False):
        """Run over ``src`` and write the annotated video to ``out``."""
        for _ in self.run(src, save=out, show=show):
            pass
        return out
