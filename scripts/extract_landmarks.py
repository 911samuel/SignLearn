"""
Landmark extraction pipeline: webcam → MediaPipe → (30, 126) NumPy sequence.

Usage:
    python scripts/extract_landmarks.py              # saves to default OUTPUT_PATH
    python scripts/extract_landmarks.py --out data/processed/my_sample.npy
    python scripts/extract_landmarks.py --out data/processed/hello/001.npy --frames 30

Output format (aligns with backend WebSocket / training pipeline):
    shape  : (SEQUENCE_LEN, FEATURE_DIM) = (30, 126)
    layout : per-frame [left_hand(63) | right_hand(63)]
    values : normalised x,y,z in [0, 1] from MediaPipe
    missing: zero-padded rows when no hand is detected

Uses MediaPipe Tasks API (mediapipe >= 0.10).
Requires: models/hand_landmarker.task
"""

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from backend.data.constants import FEATURE_DIM, HAND_DIM, SEQUENCE_LEN
from backend.data.extract import ensure_model

SEQUENCE_LENGTH = SEQUENCE_LEN   # frames per sample (~1 s at 30 FPS)
LANDMARK_DIM    = HAND_DIM       # 63 — per-hand feature width (21 × 3)
TWO_HAND_DIM    = FEATURE_DIM    # 126 — both-hands feature width
OUTPUT_PATH     = "data/processed/sample.npy"
MODEL_PATH      = _REPO_ROOT / "models" / "hand_landmarker.task"

# Landmark connections for drawing
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


def extract_row(landmarks) -> np.ndarray:
    """Flatten 21 MediaPipe Task landmarks into a (63,) float32 array."""
    return np.array(
        [[lm.x, lm.y, lm.z] for lm in landmarks],
        dtype=np.float32,
    ).flatten()


def collect_sequence(cap, landmarker, sequence_length: int, start_ms: int) -> np.ndarray:
    """Capture `sequence_length` frames and return shape (sequence_length, 126).

    Each frame is laid out as [left_hand(63) | right_hand(63)] using MediaPipe's
    handedness label. Missing hands are zero-padded.
    """
    frames = []
    prev_time = time.time()
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    while len(frames) < sequence_length:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        timestamp_ms = int(time.time() * 1000) - start_ms
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        left  = np.zeros(LANDMARK_DIM, dtype=np.float32)
        right = np.zeros(LANDMARK_DIM, dtype=np.float32)
        for landmarks, handedness_list in zip(result.hand_landmarks, result.handedness):
            label = handedness_list[0].category_name.lower()  # "left" or "right"
            row_per_hand = extract_row(landmarks)
            if label == "left":
                left = row_per_hand
            else:
                right = row_per_hand
            _draw_landmarks(frame, landmarks, w, h)

        row = np.concatenate([left, right], dtype=np.float32)
        frames.append(row)

        now = time.time()
        fps = 1.0 / (now - prev_time) if (now - prev_time) > 0 else 0.0
        prev_time = now

        cv2.putText(
            frame,
            f"Frame {len(frames)}/{sequence_length}  FPS:{fps:.1f}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2,
        )
        cv2.imshow("SignLearn – Landmark Extraction (q to abort)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Aborted by user.")
            break

    return np.array(frames, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="Extract hand landmarks to .npy")
    parser.add_argument("--out", default=OUTPUT_PATH, help="Output .npy path")
    parser.add_argument("--frames", type=int, default=SEQUENCE_LENGTH,
                        help="Number of frames to capture")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model_path = ensure_model(MODEL_PATH)
    base_options = mp.tasks.BaseOptions(model_asset_path=str(model_path))
    options = HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam.")

    print(f"Collecting {args.frames} frames — show your hand to the camera.")
    start_ms = int(time.time() * 1000)

    try:
        with HandLandmarker.create_from_options(options) as landmarker:
            data = collect_sequence(cap, landmarker, args.frames, start_ms)
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if data.shape[0] < args.frames:
        print(f"Warning: only captured {data.shape[0]}/{args.frames} frames.")

    np.save(str(out_path), data)
    print(f"Saved {data.shape} to {out_path}")
    print(f"NaNs present: {np.isnan(data).any()}")


if __name__ == "__main__":
    main()
