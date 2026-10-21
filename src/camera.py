
import cv2


class Camera:
    def __init__(self, camera_index: int = 0):
        self.capture = cv2.VideoCapture(
            camera_index,
            cv2.CAP_DSHOW
        )

        if not self.capture.isOpened():
            self.capture.release()
            raise RuntimeError(
                "Could not open webcam. Check camera permissions "
                "or try another camera index."
            )

        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def read(self):
        success, frame = self.capture.read()

        if not success:
            raise RuntimeError("Could not read a webcam frame.")

        return frame

    def release(self):
        self.capture.release()
