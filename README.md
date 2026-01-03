# TACTUS Gesture Recognizer (MediaPipe + OpenCV)

A lightweight, real-time hand-gesture recognition framework built on **MediaPipe HandLandmarker** and **OpenCV**. It captures webcam frames, extracts hand landmarks, classifies gestures (single-hand + two-hand), smooths output to reduce flicker, and optionally fires callbacks when gestures are detected.

This repo/script is designed to be used either as:
- a **standalone demo** (webcam window + overlays), or
- a **drop-in gesture module** you can import into a larger app (UI control, 3D manipulation, shortcuts, etc.).

---

## Features

- **Real-time webcam tracking** with OpenCV
- **MediaPipe Tasks HandLandmarker** in `VIDEO` mode (monotonic timestamps handled)
- Gesture detection with **priority ordering**:
  1) Two-hand gestures (higher priority)
  2) Single-hand gestures (lower priority)
- **Temporal smoothing** of gesture labels to reduce “flicker”
- **Confidence score** for each detected gesture
- **Callback hooks** per gesture type (trigger actions cleanly)
- Built-in debug overlays:
  - landmark drawing
  - gesture label + confidence bar

---

## Gesture Vocabulary

Implemented `GestureType` values:

- `NONE` — no gesture detected
- `PINCH` — thumb + index close (select/hold; precursor to rotate)
- `GRAB` — closed fist (move/drag)
- `POINT` — index extended (select/create)
- `SPREAD` — fingers spread (expand view)
- `ROTATE` — single-hand pinch+drag OR two-hand rotation (rotate object)
- `SWIPE_LEFT` — quick left motion (undo)
- `SWIPE_RIGHT` — quick right motion (redo)
- `TWO_PINCH` — both hands pinching (precise scale)

> Detection order matters: two-hand gestures are evaluated first to prevent conflicts.

---

## Requirements

- Python 3.10+ recommended
- A webcam

Python packages:
- `opencv-python`
- `mediapipe`
- `numpy`

---

## Install

```bash
pip install opencv-python mediapipe numpy
```

---

## Model File (Required)

This project expects a MediaPipe Tasks model file named:

**`hand_landmarker.task`**

By default, the script looks for it **in the same folder as the Python file**.

Place it here:
```
TACTUS_Gesture_Recognizer.py
hand_landmarker.task
```

If you want it elsewhere, pass a custom `model_path` when you initialize the recognizer.

---

## Run the Demo

```bash
python TACTUS_Gesture_Recognizer.py
```

Controls:
- Press **`q`** to quit.

You’ll see:
- webcam feed
- landmarks overlay
- gesture name + confidence bar
- detected gestures printed to console

---

## Use as a Library (Import / Integrate)

Core class: **`Gesture_Recognizer`**

Minimal integration example:

```python
import time
import cv2
from TACTUS_Gesture_Recognizer import Gesture_Recognizer, GestureType

cap = cv2.VideoCapture(0)
gr = Gesture_Recognizer()  # or Gesture_Recognizer("path/to/hand_landmarker.task")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    ts = time.time()
    state, results = gr.process_frame(frame, ts)

    if state.gesture != GestureType.NONE:
        print(state.gesture.name, state.confidence)

    cv2.imshow("TACTUS", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
```

Returned values:
- `state` → a `GestureState` containing the selected gesture + confidence (and other internal state)
- `results` → raw MediaPipe landmarker results (useful for debugging/visualization)

---

## Gesture Callbacks (Trigger Actions)

You can register callbacks per gesture:

```python
from TACTUS_Gesture_Recognizer import Gesture_Recognizer, GestureType

def on_pinch(state):
    print("PINCH detected!", state.confidence)

gr = Gesture_Recognizer()
gr.register_callback(GestureType.PINCH, on_pinch)
```

Callbacks fire when the recognizer decides that gesture is active (based on the internal smoothing + transition logic).

---

## Tuning / Thresholds

Gesture behavior is controlled by constants inside `Gesture_Recognizer`, including:
- swipe velocity thresholds
- displacement / max time windows
- smoothing factors
- two-hand rotate sensitivity

Look for the tuning block near the top of the class (examples include `SWIPE_VEL_START`, `SWIPE_MIN_DISPLACEMENT`, `TWO_HAND_ROTATE_MIN`, etc.).

Tip: small changes can drastically affect UX depending on:
- camera FOV
- how far your hands are from the camera
- lighting/background

---

## Troubleshooting

**“Missing model file: hand_landmarker.task”**
- Put `hand_landmarker.task` in the same directory as the script, or pass `model_path=` explicitly.

**Webcam won’t open**
- Close other apps using the camera (Zoom, FaceTime, browsers).
- Try changing the camera index: `cv2.VideoCapture(1)`.

**Gestures feel jittery**
- Increase smoothing / confidence thresholds slightly.
- Improve lighting and keep hands within frame.

---

## Roadmap Ideas (Optional)

- Add more gestures (zoom, double-tap pinch, palm open/close)
- Per-gesture cooldowns / debouncing for “discrete” actions
- Higher-level “mode” layer (e.g., pinch = select, pinch+move = drag, pinch+rotate = rotate)
- Multi-camera / depth integration (if available)

---

## License

MIT License — see `LICENSE` for details.
