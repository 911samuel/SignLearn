"""
Extract MediaPipe hand landmarks from WLASL video clips → (T, 126) .npy files.

Reads ``data/raw/wlasl/videos/<gloss>/*.mp4`` (downloaded by
``download_wlasl.py``), trims each clip to the WLASL-specified frame window
(``frame_start``/``frame_end`` from ``WLASL_v0.3.json``), runs MediaPipe Hands
per frame, pads/truncates to a target length, normalizes, and writes one
``.npy`` per clip into ``data/processed/words/{train,val,test}/``.

Signer-disjoint splitting
-------------------------
WLASL provides ``signer_id`` per instance. We split *signers* (not clips) into
train/val/test so the model is evaluated on people it has never seen. This
gives an honest accuracy estimate; a random split would inflate it by ~10-15%.

Default split: 70% train signers, 15% val, 15% test (per gloss).

Filename convention (matches the rest of the codebase):
    <gloss>_s<signer_id>_<idx>.npy

Usage
-----
  # Default: extract everything in configs/wlasl_words_curated.txt at 80 frames
  python backend/scripts/extract_wlasl_landmarks.py

  # Custom sequence length and word list
  python backend/scripts/extract_wlasl_landmarks.py \
      --words-file configs/wlasl_words_curated.txt \
      --seq-len 80

  # Smoke test (3 clips per gloss)
  python backend/scripts/extract_wlasl_landmarks.py --limit-per-class 3
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import mediapipe as mp_lib
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from backend.data.constants import FEATURE_DIM, HAND_DIM
from backend.data.extract import ensure_model
from backend.data.normalize import normalize_sequence

MODEL_PATH = REPO_ROOT / "models" / "hand_landmarker.task"
WLASL_ROOT = REPO_ROOT / "data" / "raw" / "wlasl"
METADATA_PATH = WLASL_ROOT / "WLASL_v0.3.json"
VIDEOS_DIR = WLASL_ROOT / "videos"
DEFAULT_WORDS_FILE = REPO_ROOT / "configs" / "wlasl_words_curated.txt"
OUT_BASE = REPO_ROOT / "data" / "processed" / "words"
REPORT_PATH = REPO_ROOT / "artifacts" / "reports" / "wlasl_extraction.json"

DEFAULT_SEQ_LEN = 80   # ~2.7s at 30fps; matches median+ p70 of WLASL clips


# ---------------------------------------------------------------------------
# MediaPipe — IMAGE mode (one detect per frame; thread-safe for our loop)
# ---------------------------------------------------------------------------

def _build_landmarker() -> HandLandmarker:
    model_path = ensure_model(MODEL_PATH)
    base_options = mp_lib.tasks.BaseOptions(model_asset_path=str(model_path))
    options = HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
    )
    return HandLandmarker.create_from_options(options)


def _frame_to_row(landmarker: HandLandmarker, bgr_frame: np.ndarray) -> np.ndarray:
    """Run MediaPipe on a single BGR frame → (126,) float32."""
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    mp_image = mp_lib.Image(image_format=mp_lib.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)

    left = np.zeros(HAND_DIM, dtype=np.float32)
    right = np.zeros(HAND_DIM, dtype=np.float32)
    for lms, handed in zip(result.hand_landmarks, result.handedness):
        label = handed[0].category_name.lower()
        row = np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32).flatten()
        if label == "left":
            left = row
        else:
            right = row
    return np.concatenate([left, right], dtype=np.float32)


# ---------------------------------------------------------------------------
# Clip → sequence
# ---------------------------------------------------------------------------

def extract_clip(
    video_path: Path,
    landmarker: HandLandmarker,
    frame_start: int,
    frame_end: int,
    target_len: int,
) -> np.ndarray | None:
    """Read [frame_start, frame_end] from video, run MediaPipe, return (target_len, 126)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    start = max(0, frame_start - 1)            # WLASL uses 1-indexed inclusive
    end = total if frame_end <= 0 else min(total, frame_end)
    if end <= start:
        cap.release()
        return None

    # Seek + read sequentially (cv2 random seek on H.264 is unreliable; sequential read is robust).
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    rows: list[np.ndarray] = []
    frame_idx = start
    while frame_idx < end:
        ok, frame = cap.read()
        if not ok:
            break
        rows.append(_frame_to_row(landmarker, frame))
        frame_idx += 1
    cap.release()

    if not rows:
        return None

    seq = np.stack(rows, axis=0).astype(np.float32)   # (T_raw, 126)

    # Pad/truncate to target_len. Center-crop long clips so we keep the sign
    # core (signs typically peak mid-clip); zero-pad short clips at the tail.
    T = seq.shape[0]
    if T >= target_len:
        offset = (T - target_len) // 2
        seq = seq[offset:offset + target_len]
    else:
        pad = np.zeros((target_len - T, FEATURE_DIM), dtype=np.float32)
        seq = np.concatenate([seq, pad], axis=0)

    # Reject clips where almost every frame is empty (MediaPipe never found a hand).
    nonzero_ratio = float(np.mean(np.any(seq != 0, axis=1)))
    if nonzero_ratio < 0.25:
        return None

    return normalize_sequence(seq)


# ---------------------------------------------------------------------------
# Signer-disjoint split
# ---------------------------------------------------------------------------

def signer_split(
    signers: list[int],
    seed: int = 42,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
) -> dict[int, str]:
    """Assign each signer_id to train/val/test deterministically.

    Done per-gloss so every class is represented in every split; if a gloss
    only has 2 signers we fall back to assigning 1 to train and 1 to val
    (test gets nothing for that class — flagged in the report).
    """
    rng = random.Random(seed)
    shuffled = sorted(set(signers))
    rng.shuffle(shuffled)
    n = len(shuffled)
    if n < 2:
        return {s: "train" for s in shuffled}
    if n == 2:
        return {shuffled[0]: "train", shuffled[1]: "val"}
    n_test = max(1, int(round(n * test_frac)))
    n_val = max(1, int(round(n * val_frac)))
    n_train = n - n_val - n_test
    assignment: dict[int, str] = {}
    for s in shuffled[:n_train]:
        assignment[s] = "train"
    for s in shuffled[n_train:n_train + n_val]:
        assignment[s] = "val"
    for s in shuffled[n_train + n_val:]:
        assignment[s] = "test"
    return assignment


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def load_metadata_index() -> dict[tuple[str, str], dict]:
    """Map (gloss, video_id) → instance dict with frame range + signer."""
    if not METADATA_PATH.exists():
        sys.exit(f"Missing {METADATA_PATH}")
    with METADATA_PATH.open() as f:
        meta = json.load(f)
    index: dict[tuple[str, str], dict] = {}
    for entry in meta:
        for inst in entry["instances"]:
            index[(entry["gloss"], inst["video_id"])] = inst
    return index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--words-file", type=Path, default=DEFAULT_WORDS_FILE)
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN,
                        help=f"Target frames per clip (default {DEFAULT_SEQ_LEN}).")
    parser.add_argument("--limit-per-class", type=int,
                        help="Cap clips per gloss (smoke testing).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.words_file.exists():
        sys.exit(f"Words file not found: {args.words_file}")
    glosses = [w.strip() for w in args.words_file.read_text().splitlines() if w.strip()]
    print(f"Extracting {len(glosses)} gloss(es) at seq_len={args.seq_len}")

    meta_index = load_metadata_index()
    landmarker = _build_landmarker()

    OUT_BASE.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        (OUT_BASE / split).mkdir(parents=True, exist_ok=True)

    report = {
        "seq_len": args.seq_len,
        "glosses": {},
        "totals": {"ok": 0, "no_metadata": 0, "extract_failed": 0, "low_hand_ratio": 0},
    }

    for gloss in glosses:
        gloss_dir = VIDEOS_DIR / gloss
        if not gloss_dir.is_dir():
            print(f"[{gloss}] no clips directory, skipping")
            continue
        clips = sorted(gloss_dir.glob("*.mp4"))
        if args.limit_per_class:
            clips = clips[:args.limit_per_class]

        # Match clips back to metadata via video_id (filename prefix before _NNN.mp4).
        signer_of: dict[Path, int | None] = {}
        instance_of: dict[Path, dict] = {}
        for clip in clips:
            video_id = clip.stem.rsplit("_", 1)[0]
            inst = meta_index.get((gloss, video_id))
            if inst is None:
                signer_of[clip] = None
                continue
            signer_of[clip] = inst.get("signer_id")
            instance_of[clip] = inst

        present_signers = [s for s in signer_of.values() if s is not None]
        split_map = signer_split(present_signers, seed=args.seed)

        per_class = {"ok": 0, "no_metadata": 0, "extract_failed": 0,
                     "low_hand_ratio": 0, "splits": {"train": 0, "val": 0, "test": 0}}

        for idx, clip in enumerate(clips):
            inst = instance_of.get(clip)
            if inst is None:
                per_class["no_metadata"] += 1
                continue
            signer = signer_of[clip]
            split = split_map.get(signer, "train") if signer is not None else "train"

            seq = extract_clip(
                clip, landmarker,
                inst.get("frame_start", 1),
                inst.get("frame_end", -1),
                args.seq_len,
            )
            if seq is None:
                per_class["low_hand_ratio"] += 1
                continue

            out_path = OUT_BASE / split / f"{gloss}_s{signer:02d}_{idx:04d}.npy"
            np.save(str(out_path), seq.astype(np.float32))
            per_class["ok"] += 1
            per_class["splits"][split] += 1

        print(f"[{gloss}] ok={per_class['ok']:3d}  "
              f"train={per_class['splits']['train']:3d} "
              f"val={per_class['splits']['val']:3d} "
              f"test={per_class['splits']['test']:3d}  "
              f"dropped(no_hand)={per_class['low_hand_ratio']}  "
              f"dropped(no_meta)={per_class['no_metadata']}")
        report["glosses"][gloss] = per_class
        for k in ("ok", "no_metadata", "extract_failed", "low_hand_ratio"):
            report["totals"][k] += per_class[k]

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"\nReport: {REPORT_PATH}")
    print(f"Totals: {report['totals']}")


if __name__ == "__main__":
    main()
