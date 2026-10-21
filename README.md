# Facial Expression Recognition by MoSaZeaL

A modular, local webcam application for detecting faces and estimating visible facial-expression categories using Python 3.14.8, OpenCV, and pretrained ONNX models.

> **Important:** Expression predictions are model estimates based on visible facial features. They do not reliably reveal a person's internal emotional state.

## Features

- Real-time webcam preview.
- Face detection using OpenCV Zoo's YuNet model.
- Facial-expression classification using a pretrained ONNX model.
- Modular code separated into camera, face detection, classification, and application entry point.
- Local inference after dependencies and model files have been downloaded.
- Designed to run on CPU; GPU acceleration is optional and not required.

## Requirements

- Windows 10/11
- Python **3.14.8**
- A webcam
- Internet access for initial package/model downloads
- OpenCV-compatible ONNX model files

## Project structure

```text
FacialExpressionApp/
├── models/
│   ├── face_detection_yunet.onnx
│   └── expression_recognition.onnx
├── src/
│   ├── camera.py
│   ├── face_detector.py
│   ├── expression_classifier.py
│   └── main.py
├── requirements.txt
└── README.md
```

## 1. Create and activate a virtual environment

Run these commands in PowerShell from the project directory:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version
python -m pip install --upgrade pip setuptools wheel
```

The Python version should report `Python 3.14.8`.

If PowerShell blocks virtual-environment activation, you can run the environment's Python directly:

```powershell
.\.venv\Scripts\python.exe --version
```

## 2. Install dependencies

Install OpenCV and NumPy using prebuilt wheels:

```powershell
python -m pip install --only-binary=:all: numpy opencv-python
```

Verify the installation:

```powershell
python -c "import numpy, cv2; print('NumPy:', numpy.__version__); print('OpenCV:', cv2.__version__)"
```

The current implementation uses OpenCV's DNN module to load ONNX models. `onnxruntime` is not required for that implementation.

### `requirements.txt`

```text
numpy
opencv-python
```

For reproducible installations, record the tested package versions after the application works:

```powershell
python -m pip freeze > requirements-lock.txt
```

## 3. Download the pretrained models

Create the model directory if it does not exist:

```powershell
New-Item -ItemType Directory -Force .\models
```

### Face detection: YuNet

Official project and model instructions:

- https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet

Expected local filename:

`models/face_detection_yunet.onnx`

### Facial-expression recognition

Official project and demo:

- https://github.com/opencv/opencv_zoo/tree/main/models/facial_expression_recognition
- https://github.com/opencv/opencv_zoo/blob/main/models/facial_expression_recognition/demo.py

Expected local filename:

`models/expression_recognition.onnx`

Download links and model revisions can change. Prefer the official repository's model page and download the actual `.onnx` asset. Do not save an HTML error page with an `.onnx` extension.

Check downloaded files:

```powershell
Get-ChildItem .\models | Select-Object Name, Length
```

Check the exact path from Python:

```powershell
python -c "from pathlib import Path; p=Path('models/face_detection_yunet.onnx'); print('Working directory:', Path.cwd()); print('Resolved path:', p.resolve()); print('Exists:', p.is_file())"
```

The app resolves model paths relative to the project root calculated from `src/main.py`, not from the current working directory.

## 4. Run the application

From the project root:

```powershell
python src\main.py
```

The webcam window should display detected faces and expression labels once both models are present and their inference code matches the downloaded model interfaces. Press **Q** or **Esc** to exit.

## 5. Architecture

1. **Camera (`camera.py`)** — opens the webcam and returns frames.
2. **Face detector (`face_detector.py`)** — detects faces and returns bounding boxes and landmarks.
3. **Expression classifier (`expression_classifier.py`)** — aligns each detected face, preprocesses it, and runs the classifier.
4. **Application (`main.py`)** — coordinates the modules, draws results, and displays FPS.

## Troubleshooting

### `Face detection model not found`

1. Confirm the filename is exactly `face_detection_yunet.onnx`.
2. Ensure it is inside the project's `models` directory.
3. Check for accidental extensions such as `.onnx.onnx` or `.onnx.txt`.
4. Run:

   ```powershell
   Get-ChildItem .\models
   python -c "from pathlib import Path; p=Path(r'D:\Projects\Python\Emotion-Recognition\Emotion-Recognition\models\face_detection_yunet.onnx'); print(p.is_file(), p)"
   ```

   Replace the example path if your project is stored elsewhere.

### `pkgutil.ImpImporter` / `fer-pytorch` installation error

The older `fer-pytorch` package may rely on build dependencies that are incompatible with Python 3.14. This project does **not** require `fer-pytorch`. Use OpenCV and compatible pretrained ONNX models instead.

### Model input/output errors

ONNX files do not all share the same input names, tensor shapes, color ordering, normalization, or output format. Use the official demo associated with the exact model revision. Do not assume an input name such as `data` or an output name such as `label` without checking the model's example or metadata.

### Webcam will not open

- Check Windows camera privacy permissions.
- Close other applications using the camera.
- Try a different camera index in `Camera(camera_index=0)`.
- If necessary, try a different OpenCV capture backend.

### Performance

Start with CPU inference. The NVIDIA Quadro P1000 may not benefit from every current GPU inference package/version combination. Add GPU acceleration only after the CPU version is working and the required drivers/runtime are verified.

## Current status and next steps

- [ ] Verify Python 3.14.8 virtual environment.
- [ ] Install OpenCV and NumPy.
- [ ] Download and validate both ONNX models.
- [ ] Confirm face detection works.
- [ ] Adapt expression preprocessing/inference to the official demo for the selected model.
- [ ] Test live webcam performance.
- [ ] Optionally add a GUI, temporal smoothing, configurable camera selection, and model confidence display.

## License and responsible use

Review the licenses and usage terms of each pretrained model and dataset before redistribution or commercial use. This application is intended as a technical demonstration. Do not use expression labels as a definitive assessment of someone's feelings, personality, intent, or mental state.
