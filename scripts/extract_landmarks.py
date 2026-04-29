"""
Landmark extraction pipeline: webcam → MediaPipe → (30, 63) NumPy sequence.

Usage:
    python scripts/extract_landmarks.py              # saves to default OUTPUT_PATH
    python scripts/extract_landmarks.py --out data/processed/my_sample.npy

Output format (aligns with Member B / frontend contract):
    shape  : (SEQUENCE_LENGTH, LANDMARK_DIM) = (30, 63)
    values : normalised x,y,z in [0, 1] from MediaPipe
    missing: zero-padded rows when no hand is detected
"""

import argparse
import os
import time
import cv2
import numpy as np
from mediapipe.python.solutions import hands as mp_hands
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import drawing_styles as mp_drawing_styles

SEQUENCE_LENGTH = 30   # frames per sample (~1 s at 30 FPS)
LANDMARK_DIM    = 63   # 21 landmarks × 3 coords (x, y, z)
OUTPUT_PATH     = "data/processed/sample.npy"


def extract_row(hand_landmarks) -> np.ndarray:
    """Flatten 21 MediaPipe landmarks into a (63,) array."""
    return np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
        dtype=np.float32,
    ).flatten()


def collect_sequence(cap, hands, sequence_length: int) -> np.ndarray:
    """Capture `sequence_length` frames and return shape (sequence_length, 63)."""
    frames = []
    prev_time = time.time()

    while len(frames) < sequence_length:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = hands.process(rgb)
        rgb.flags.writeable = True
        frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            row = extract_row(results.multi_hand_landmarks[0])
            mp_drawing.draw_landmarks(
                frame,
                results.multi_hand_landmarks[0],
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )
        else:
            row = np.zeros(LANDMARK_DIM, dtype=np.float32)

        frames.append(row)

        now = time.time()
        fps = 1.0 / (now - prev_time) if (now - prev_time) > 0 else 0.0
        prev_time = now

        cv2.putText(
            frame,
            f"Frame {len(frames)}/{sequence_length}  FPS:{fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
        )
        cv2.imshow("SignLearn – Landmark Extraction (q to abort)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Aborted by user.")
            break

    return np.array(frames, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="Extract hand landmarks to .npy")
    parser.add_argument("--out", default=OUTPUT_PATH, help="Output .npy path")
    parser.add_argument(
        "--frames", type=int, default=SEQUENCE_LENGTH, help="Number of frames to capture"
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam.")

    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )

    print(f"Collecting {args.frames} frames — show your hand to the camera.")
    try:
        data = collect_sequence(cap, hands, args.frames)
    finally:
        hands.close()
        cap.release()
        cv2.destroyAllWindows()

    if data.shape[0] < args.frames:
        print(f"Warning: only captured {data.shape[0]}/{args.frames} frames.")

    np.save(args.out, data)
    print(f"Saved {data.shape} to {args.out}")
    print(f"NaNs present: {np.isnan(data).any()}")


if __name__ == "__main__":
    main()
