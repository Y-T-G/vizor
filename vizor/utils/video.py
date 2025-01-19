import cv2

class VideoCapture:
    def __init__(self, cap):
        self.cap = cap
        self.length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __iter__(self):
        for i in range(self.length):
            yield self.cap.read()

    def __del__(self):
        self.cap.release()