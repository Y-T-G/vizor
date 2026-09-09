"""Reading and writing video without the usual OpenCV boilerplate."""

import cv2

__all__ = ["Video", "Writer"]


class Video:
    """Iterate the frames of a file, a camera index, or an RTSP url.

    Iteration stops on the first failed read, so a wrong frame count in the
    container header cannot cut the stream short or hand you empty frames.
    """

    def __init__(self, src):
        self.src = src
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise OSError(f"cannot open {src!r}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                     int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    def __iter__(self):
        while True:
            ok, frame = self.cap.read()
            if not ok:
                return
            yield frame

    def __len__(self):
        return max(0, self.n)

    def close(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class Writer:
    """Write frames to a video file. The size is taken from the first frame."""

    def __init__(self, path, fps=30.0, fourcc="mp4v"):
        self.path = str(path)
        self.fps = float(fps) or 30.0
        self.fourcc = fourcc
        self.out = None

    def write(self, img):
        if self.out is None:
            h, w = img.shape[:2]
            code = cv2.VideoWriter_fourcc(*self.fourcc)
            self.out = cv2.VideoWriter(self.path, code, self.fps, (w, h))
            if not self.out.isOpened():
                raise OSError(f"cannot write {self.path!r} with fourcc {self.fourcc!r}")
        self.out.write(img)

    def close(self):
        if self.out is not None:
            self.out.release()
            self.out = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
