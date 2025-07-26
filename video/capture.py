import cv2

class VideoStream:
    def __init__(self, source=0):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise Exception("Could not open video source")

    def capture(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            yield frame

    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()