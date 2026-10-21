
from pathlib import Path
import cv2


class FaceDetector:
    def __init__(self, model_path: str):
        path = Path(model_path)

        if not path.is_file():
            raise FileNotFoundError(
                f"Face detection model not found: {path}"
            )

        self.detector = cv2.FaceDetectorYN.create(
            str(path),
            "",
            (320, 320),
            0.6,   # Detection confidence threshold
            0.3,   # Non-maximum suppression threshold
            5000
        )

    def detect(self, frame):
        height, width = frame.shape[:2]

        self.detector.setInputSize((width, height))
        _, faces = self.detector.detect(frame)

        if faces is None:
            return []

        return faces

