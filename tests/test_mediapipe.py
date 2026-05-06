"""
Webcam smoke test: MediaPipe hand tracking with FPS overlay.
Run with: python tests/test_mediapipe.py
Press 'q' to quit.
Success criteria: >=15 FPS stable, landmarks visible on hand.
Appends a result line to docs/hardware.md on exit.

Uses MediaPipe Tasks API (mediapipe >= 0.10).
Requires: models/hand_landmarker.task  (downloaded via scripts/download_model.py)
"""

import time
from datetime import datetime
from pathlib import Path

import sys

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from backend.data.extract import ensure_model

HARDWARE_DOC = _REPO_ROOT / "docs" / "hardware.md"
MODEL_PATH   = _REPO_ROOT / "models" / "hand_landmarker.task"

# Landmark connections for drawing (index pairs)
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17),
]


def _draw_landmarks(frame, landmarks, w, h):
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 200, 0), 2)
    for pt in pts:
        cv2.circle(frame, pt, 4, (0, 255, 0), -1)


def _append_webcam_result(width: int, height: int, avg_fps: float) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    line = (
        f"\n## Webcam Test ({timestamp})\n"
        f"- **Resolution**: {width}×{height}\n"
        f"- **Average FPS**: {avg_fps:.1f}\n"
        f"- **Pass**: {'Yes' if avg_fps >= 15 else 'No — below 15 FPS threshold'}\n"
    )
    with open(HARDWARE_DOC, "a") as f:
        f.write(line)
    print(f"Result appended to {HARDWARE_DOC}")


def main():
    model_path = ensure_model(MODEL_PATH)
    base_options = mp.tasks.BaseOptions(model_asset_path=str(model_path))
    options = HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam — check device index or permissions.")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Webcam resolution: {width}×{height}")
    print("Press 'q' to quit and save results.")

    prev_time = time.time()
    fps_samples: list[float] = []
    start_ms = int(time.time() * 1000)

    try:
        with HandLandmarker.create_from_options(options) as landmarker:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame.")
                    break

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                timestamp_ms = int(time.time() * 1000) - start_ms
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                if result.hand_landmarks:
                    _draw_landmarks(frame, result.hand_landmarks[0], width, height)

                now = time.time()
                fps = 1.0 / (now - prev_time) if (now - prev_time) > 0 else 0.0
                prev_time = now
                fps_samples.append(fps)

                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                cv2.imshow("SignLearn – MediaPipe Test (q to quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if fps_samples:
        avg_fps = (sum(fps_samples[5:]) / len(fps_samples[5:])
                   if len(fps_samples) > 5 else sum(fps_samples) / len(fps_samples))
        print(f"Average FPS: {avg_fps:.1f} (pass: {avg_fps >= 15})")
        _append_webcam_result(width, height, avg_fps)


if __name__ == "__main__":
    main()
