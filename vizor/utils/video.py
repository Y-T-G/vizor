import cv2

class IterableVideoCapture:
    def __init__(self, file):
        self.cap = cv2.VideoCapture(file)
        self.length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __iter__(self):
        for i in range(self.length):
            yield self.cap.read()

    def __del__(self):
        self.cap.release()