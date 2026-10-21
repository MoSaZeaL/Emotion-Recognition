import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QIcon, QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from camera import Camera
from face_detector import FaceDetector
from expression_classifier import ExpressionClassifier


PROJECT_ROOT = Path(__file__).resolve().parent.parent

FACE_MODEL = (
    PROJECT_ROOT / "models" / "face_detection_yunet.onnx"
)
EXPRESSION_MODEL = (
    PROJECT_ROOT / "models" / "expression_recognition.onnx"
)
HAND_MODEL = (
    PROJECT_ROOT / "models" / "hand_landmarker.task"
)

# Icon file — adjust the name/path to match your file
ICON_PATH = PROJECT_ROOT / "assets" / "icon.png"


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Facial Expression Recognition")
        self.resize(960, 720)

        # --- Central widget & layout ---
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        self.video_label = QLabel("Starting camera...")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        layout.addWidget(self.video_label)

        # --- Camera ---
        self.camera = Camera()

        # --- Face detector / expression classifier ---
        self.detector = FaceDetector(str(FACE_MODEL))
        self.classifier = ExpressionClassifier(str(EXPRESSION_MODEL))

        # --- MediaPipe HandLandmarker (new Tasks API) ---
        base_options = python.BaseOptions(
            model_asset_path=str(HAND_MODEL)
        )
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.7,
            min_tracking_confidence=0.7,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)

        # --- FPS state ---
        self.previous_time = time.perf_counter()
        self.fps = 0.0
        self._last_face_count = 0

        # --- Timer drives the frame loop ---
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(15)  # ~66 FPS cap

    # ---------------------------------------------------------------

    def update_frame(self):
        frame = self.camera.read()
        if frame is None:
            return

        frame = self._process_hands(frame)
        frame = self._process_faces(frame)
        frame = self._draw_fps(frame)

        self._display(frame)

    # ---------------------------------------------------------------

    def _process_hands(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        timestamp_ms = int(time.perf_counter() * 1000)
        hand_result = self.landmarker.detect_for_video(
            mp_image, timestamp_ms
        )

        if not hand_result.hand_landmarks:
            return frame

        h, w, _ = frame.shape

        for index, hand_landmarks in enumerate(
            hand_result.hand_landmarks
        ):
            # Draw 21 keypoints
            for lm in hand_landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)

            finger_count = 0
            tip_ids = [4, 8, 12, 16, 20]
            joint_ids = [3, 6, 10, 14, 18]

            if index < len(hand_result.handedness):
                handedness = (
                    hand_result.handedness[index][0].display_name
                )
            else:
                handedness = "Right"

            thumb_tip = hand_landmarks[tip_ids[0]]
            thumb_ip = hand_landmarks[joint_ids[0]]

            if handedness == "Right":
                if thumb_tip.x < thumb_ip.x:
                    finger_count += 1
            else:
                if thumb_tip.x > thumb_ip.x:
                    finger_count += 1

            for i in range(1, 5):
                tip = hand_landmarks[tip_ids[i]]
                joint = hand_landmarks[joint_ids[i]]
                if tip.y < joint.y:
                    finger_count += 1

            wrist = hand_landmarks[0]
            wrist_x = int(wrist.x * w)
            wrist_y = int(wrist.y * h)
            text_y = max(25, wrist_y - 30)

            cv2.putText(
                frame,
                f"Fingers: {finger_count}",
                (wrist_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA
            )

        return frame

    # ---------------------------------------------------------------

    def _process_faces(self, frame):
        faces = self.detector.detect(frame)

        for face in faces:
            x, y, width, height = map(int, face[:4])
            label, confidence = self.classifier.predict(frame, face)

            x = max(0, x)
            y = max(0, y)

            cv2.rectangle(
                frame,
                (x, y),
                (x + width, y + height),
                (0, 220, 0),
                2
            )

            text = (
                f"{label} ({confidence:.2f})"
                if label != "Unknown"
                else label
            )
            text_y = max(25, y - 10)

            cv2.putText(
                frame,
                text,
                (x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 220, 0),
                2,
                cv2.LINE_AA
            )

        self._last_face_count = len(faces)
        return frame

    # ---------------------------------------------------------------

    def _draw_fps(self, frame):
        current_time = time.perf_counter()
        elapsed = current_time - self.previous_time
        self.previous_time = current_time

        if elapsed > 0:
            instant_fps = 1.0 / elapsed
            self.fps = (
                instant_fps if self.fps == 0
                else 0.9 * self.fps + 0.1 * instant_fps
            )

        cv2.putText(
            frame,
            f"FPS: {self.fps:.1f} | Faces: {self._last_face_count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
            cv2.LINE_AA
        )
        return frame

    # ---------------------------------------------------------------

    def _display(self, frame):
        # BGR -> RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape

        # Copy because QImage doesn't own the buffer
        bytes_per_line = ch * w
        q_img = QImage(
            rgb.data, w, h, bytes_per_line,
            QImage.Format.Format_RGB888
        ).copy()

        pix = QPixmap.fromImage(q_img)

        self.video_label.setPixmap(
            pix.scaled(
                self.video_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    # ---------------------------------------------------------------

    def closeEvent(self, event):
        self.timer.stop()
        try:
            self.camera.release()
        except Exception:
            pass
        try:
            self.landmarker.close()
        except Exception:
            pass
        event.accept()


def main():
    # --- Set Windows taskbar identity (must run before QApplication) ---
    if sys.platform == "win32":
        import ctypes
        myappid = "com.yourcompany.facialexpression.1.0"
        try:
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                myappid
            )
        except Exception:
            pass

    app = QApplication(sys.argv)

    # --- Application icon ---
    if ICON_PATH.exists():
        app.setWindowIcon(QIcon(str(ICON_PATH)))
    else:
        print(f"Warning: icon not found at {ICON_PATH}")

    try:
        window = MainWindow()
        # Window icon also (covers cases where app icon isn't inherited)
        if ICON_PATH.exists():
            window.setWindowIcon(QIcon(str(ICON_PATH)))
        window.show()
        sys.exit(app.exec())
    except (FileNotFoundError, RuntimeError) as error:
        QMessageBox.critical(None, "Application error", str(error))
        sys.exit(1)


if __name__ == "__main__":
    main()