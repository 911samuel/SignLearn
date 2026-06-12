"""
Extract MediaPipe hand-landmark sequences from ASL Citizen for a curated
list of demo words. Streams videos directly out of the zip (does not unpack
the full 45 GB archive). Uses ASL Citizen's built-in train/val/test splits.

Inputs:
  data/raw/ASL_Citizen.zip           # the downloaded dataset
  configs/asl_citizen_demo_words.txt # one gloss per line (ASL-LEX casing)

Outputs:
  data/processed/words/{train,val,test}/<gloss_lower>_sac<participant>_<idx>.npy

Each .npy is shape (80, 126) float32 — same layout as the WLASL extractor and
the train_word_model.py expects.

The `sac` prefix in the filename distinguishes ASL Citizen rows from WLASL (`s`)
and YouTube (`syt`) so the trainer can mix or partition them.

Usage:
  python backend/scripts/extract_asl_citizen_landmarks.py
  python backend/scripts/extract_asl_citizen_landmarks.py --limit-per-class 5  # smoke test
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import shutil
import sys
import tempfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import mediapipe as mp_lib
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions

REPO = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO))

from backend.data.constants import FEATURE_DIM, HAND_DIM
from backend.data.extract import ensure_model
from backend.data.normalize import normalize_sequence

ZIP = REPO / "data" / "raw" / "ASL_Citizen.zip"
WORDS = REPO / "configs" / "asl_citizen_demo_words.txt"
OUT = REPO / "data" / "processed" / "words"
REPORT = REPO / "artifacts" / "reports" / "asl_citizen_extraction.json"
MODEL_PATH = REPO / "models" / "hand_landmarker.task"

SEQ_LEN = 80


def _build_landmarker() -> HandLandmarker:
    model = ensure_model(MODEL_PATH)
    base = mp_lib.tasks.BaseOptions(model_asset_path=str(model))
    opts = HandLandmarkerOptions(
        base_options=base,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
    )
    return HandLandmarker.create_from_options(opts)


def _frame_to_row(landmarker: HandLandmarker, bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
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


def extract_video(video_path: Path, landmarker: HandLandmarker) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    rows = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rows.append(_frame_to_row(landmarker, frame))
    cap.release()
    if not rows:
        return None
    seq = np.stack(rows).astype(np.float32)
    T = seq.shape[0]
    if T >= SEQ_LEN:
        offset = (T - SEQ_LEN) // 2
        seq = seq[offset:offset + SEQ_LEN]
    else:
        pad = np.zeros((SEQ_LEN - T, FEATURE_DIM), dtype=np.float32)
        seq = np.concatenate([seq, pad], axis=0)
    nonzero = float(np.mean(np.any(seq != 0, axis=1)))
    if nonzero < 0.25:
        return None
    return normalize_sequence(seq)


def safe_signer_id(participant_id: str) -> str:
    """Normalise participant ID into a short alphanumeric token for filenames."""
    return re.sub(r"[^A-Za-z0-9]+", "", participant_id)[:8] or "unk"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-per-class", type=int,
                        help="Cap clips per class per split (smoke test).")
    parser.add_argument("--words-file", type=Path, default=WORDS,
                        help=f"Path to newline-delimited gloss list. Default: {WORDS}")
    args = parser.parse_args()

    if not ZIP.exists():
        sys.exit(f"ASL Citizen zip not found at {ZIP}")
    if not args.words_file.exists():
        sys.exit(f"Demo word list not found at {args.words_file}")

    wanted = {g.strip() for g in args.words_file.read_text().splitlines()
              if g.strip() and not g.strip().startswith("#")}
    print(f"Target classes: {len(wanted)}")

    # Build manifest: gloss → [(split, video_file, participant_id), ...]
    by_gloss: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    with zipfile.ZipFile(ZIP) as zf:
        for split in ("train", "val", "test"):
            with zf.open(f"ASL_Citizen/splits/{split}.csv") as f:
                for row in csv.DictReader(io.TextIOWrapper(f, "utf-8")):
                    g = row["Gloss"]
                    if g in wanted:
                        by_gloss[g].append((split, row["Video file"], row["Participant ID"]))

    print(f"Manifest built — {sum(len(v) for v in by_gloss.values()):,} candidate clips "
          f"across {len(by_gloss)} matched glosses.")

    landmarker = _build_landmarker()
    OUT.mkdir(parents=True, exist_ok=True)
    for s in ("train", "val", "test"):
        (OUT / s).mkdir(parents=True, exist_ok=True)

    report = {
        "seq_len": SEQ_LEN,
        "glosses": {},
        "totals": {"ok": 0, "no_hand": 0, "video_open_failed": 0, "missing_in_zip": 0},
    }

    tmp_dir = Path(tempfile.mkdtemp(prefix="aslc_"))
    print(f"Temp extraction dir: {tmp_dir}")

    try:
        with zipfile.ZipFile(ZIP) as zf:
            for gloss in sorted(by_gloss):
                entries = by_gloss[gloss]
                if args.limit_per_class:
                    # Spread the limit across splits
                    by_split = defaultdict(list)
                    for e in entries:
                        by_split[e[0]].append(e)
                    entries = []
                    for s in ("train", "val", "test"):
                        entries.extend(by_split[s][:args.limit_per_class])
                gloss_lower = gloss.lower()
                per_class = {"ok": 0, "no_hand": 0, "video_open_failed": 0,
                             "missing_in_zip": 0,
                             "splits": {"train": 0, "val": 0, "test": 0}}
                for idx, (split, video_file, pid) in enumerate(entries):
                    sid = safe_signer_id(pid)
                    out_path = OUT / split / f"{gloss_lower}_sac{sid}_{idx:04d}.npy"
                    if out_path.exists() and out_path.stat().st_size > 0:
                        per_class["ok"] += 1
                        per_class["splits"][split] += 1
                        continue  # already extracted — safe resume
                    zip_path = f"ASL_Citizen/videos/{video_file}"
                    try:
                        info = zf.getinfo(zip_path)
                    except KeyError:
                        per_class["missing_in_zip"] += 1
                        continue
                    # Extract single video to temp dir
                    tmp_video = tmp_dir / f"_tmp_{idx}.mp4"
                    with zf.open(info) as src, open(tmp_video, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    seq = extract_video(tmp_video, landmarker)
                    tmp_video.unlink(missing_ok=True)
                    if seq is None:
                        per_class["no_hand"] += 1
                        continue
                    np.save(str(out_path), seq.astype(np.float32))
                    per_class["ok"] += 1
                    per_class["splits"][split] += 1

                print(f"[{gloss:<14}] ok={per_class['ok']:3d}  "
                      f"train={per_class['splits']['train']:3d} "
                      f"val={per_class['splits']['val']:3d} "
                      f"test={per_class['splits']['test']:3d}  "
                      f"no_hand={per_class['no_hand']}")
                report["glosses"][gloss] = per_class
                for k in ("ok", "no_hand", "video_open_failed", "missing_in_zip"):
                    report["totals"][k] += per_class[k]
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2))
    print(f"\nReport: {REPORT}")
    print(f"Totals: {report['totals']}")


if __name__ == "__main__":
    main()
