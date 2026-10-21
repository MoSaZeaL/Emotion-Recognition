
from pathlib import Path

import cv2
import numpy as np


class ExpressionClassifier:
    LABELS = [
        "Angry",
        "Disgust",
        "Fearful",
        "Happy",
        "Neutral",
        "Sad",
        "Surprised"
    ]

    # Standard landmark positions for a 112x112 aligned face.
    DESTINATION_POINTS = np.array(
        [
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041]
        ],
        dtype=np.float32
    )

    def __init__(self, model_path: str):
        path = Path(model_path)

        if not path.is_file():
            raise FileNotFoundError(
                f"Expression model not found: {path}"
            )

        self.model = cv2.dnn.readNet(str(path))

    def _align_face(self, frame, face):
        # YuNet: x, y, width, height, five (x, y) landmarks, score.
        landmarks = face[4:14].reshape(5, 2).astype(np.float32)

        transform, _ = cv2.estimateAffinePartial2D(
            landmarks,
            self.DESTINATION_POINTS,
            method=cv2.LMEDS
        )

        if transform is None:
            return None

        return cv2.warpAffine(
            frame,
            transform,
            (112, 112),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT
        )

    def predict(self, frame, face):
        aligned = self._align_face(frame, face)

        if aligned is None:
            return "Unknown", 0.0

        # The model expects RGB values normalized to [-1, 1].
        rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
        image = rgb.astype(np.float32) / 255.0
        image = (image - 0.5) / 0.5

        blob = cv2.dnn.blobFromImage(image)

        self.model.setInput(blob, "data")
        scores = self.model.forward("label").reshape(-1)

        if scores.size != len(self.LABELS):
            raise RuntimeError(
                f"Unexpected model output shape: {scores.shape}"
            )

        index = int(np.argmax(scores))
        confidence = float(scores[index])

        return self.LABELS[index], confidence
