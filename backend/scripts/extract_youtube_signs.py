"""
Download isolated-word ASL sign videos from YouTube and extract landmarks.

Use this when you don't sign ASL yourself but need clips for words missing
from WLASL (e.g. ``hello``, ``thank_you``, ``please``, ``help``). Educational
channels like Bill Vicars / Lifeprint and Signing Savvy host individual-word
demos that work as training data.

Pipeline:
  1. Read configs/signs_youtube.tsv  (gloss, url, signer_tag, start, end)
  2. Download each URL with yt-dlp, trimmed via --download-sections to [start,end]
  3. Run MediaPipe Hands per frame, pad/truncate to seq_len, normalize
  4. Signer-disjoint split by signer_tag → data/processed/words/{train,val,test}

Filename convention (matches WLASL output, so both sources train together):
  <gloss>_syt<hash>_<idx>.npy
  (signer hash is deterministic per signer_tag; ``yt`` prefix distinguishes
   YouTube-sourced rows from WLASL numeric signer IDs.)

Prerequisites:
  pip install yt-dlp
  ffmpeg on PATH

Usage:
  python backend/scripts/extract_youtube_signs.py
  python backend/scripts/extract_youtube_signs.py --tsv configs/signs_youtube.tsv
  python backend/scripts/extract_youtube_signs.py --seq-len 80 --dry-run

Legal note:
  Treat sourced clips as fair-use educational research. Do not redistribute the
  raw .mp4 files; the .npy landmark sequences contain no copyrightable footage
  and are safe to keep in the dataset. Cite source channels in your final
  report's data section.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import subprocess
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
DEFAULT_TSV = REPO_ROOT / "configs" / "signs_youtube.tsv"
RAW_DIR = REPO_ROOT / "data" / "raw" / "youtube_signs"
OUT_BASE = REPO_ROOT / "data" / "processed" / "words"
REPORT_PATH = REPO_ROOT / "artifacts" / "reports" / "youtube_extraction.json"

DEFAULT_SEQ_LEN = 80


# ---------------------------------------------------------------------------
# TSV loader
# ---------------------------------------------------------------------------

def load_rows(tsv_path: Path) -> list[dict]:
    if not tsv_path.exists():
        sys.exit(f"TSV not found: {tsv_path}")
    rows = []
    with tsv_path.open() as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split("\t")]
            if len(parts) < 3:
                print(f"WARN line {lineno}: need ≥3 tab-separated fields, got {parts!r}")
                continue
            gloss, url, signer_tag = parts[0], parts[1], parts[2]
            start = float(parts[3]) if len(parts) > 3 and parts[3] else 0.0
            end = float(parts[4]) if len(parts) > 4 and parts[4] else 0.0
            rows.append({
                "gloss": gloss,
                "url": url,
                "signer_tag": signer_tag.lower(),
                "start": start,
                "end": end,
                "lineno": lineno,
            })
    return rows


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def _yt_dlp_cmd() -> list[str]:
    binary = shutil.which("yt-dlp")
    if binary:
        return [binary]
    try:
        import yt_dlp  # noqa: F401
        return [sys.executable, "-m", "yt_dlp"]
    except ImportError:
        sys.exit("yt-dlp not installed. Run: pip install yt-dlp")


def download_clip(row: dict, dest: Path) -> bool:
    """Download and trim a clip. Returns True on success."""
    if dest.exists() and dest.stat().st_size > 0:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = _yt_dlp_cmd() + [
        "--quiet", "--no-warnings", "--no-playlist",
        "-f", "mp4/best",
        "-o", str(dest),
        row["url"],
    ]
    if row["end"] > row["start"]:
        # --download-sections trims server-side when possible (much faster).
        cmd += ["--download-sections", f"*{row['start']}-{row['end']}",
                "--force-keyframes-at-cuts"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    except subprocess.TimeoutExpired:
        return False
    return result.returncode == 0 and dest.exists() and dest.stat().st_size > 0


# ---------------------------------------------------------------------------
# MediaPipe
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


def extract_clip(video_path: Path, landmarker: HandLandmarker, seq_len: int) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    rows: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rows.append(_frame_to_row(landmarker, frame))
    cap.release()
    if not rows:
        return None

    seq = np.stack(rows, axis=0).astype(np.float32)
    T = seq.shape[0]
    if T >= seq_len:
        offset = (T - seq_len) // 2
        seq = seq[offset:offset + seq_len]
    else:
        pad = np.zeros((seq_len - T, FEATURE_DIM), dtype=np.float32)
        seq = np.concatenate([seq, pad], axis=0)

    nonzero_ratio = float(np.mean(np.any(seq != 0, axis=1)))
    if nonzero_ratio < 0.25:
        return None
    return normalize_sequence(seq)


# ---------------------------------------------------------------------------
# Signer-disjoint split
# ---------------------------------------------------------------------------

def signer_hash(signer_tag: str) -> str:
    """Stable 6-char hash for filename uniqueness."""
    return hashlib.sha1(signer_tag.encode()).hexdigest()[:6]


def assign_splits(signer_tags: set[str], seed: int = 42) -> dict[str, str]:
    """Deterministically assign each signer_tag to train/val/test."""
    rng = random.Random(seed)
    tags = sorted(signer_tags)
    rng.shuffle(tags)
    n = len(tags)
    if n == 0:
        return {}
    if n == 1:
        return {tags[0]: "train"}
    if n == 2:
        return {tags[0]: "train", tags[1]: "val"}
    n_test = max(1, n // 5)
    n_val = max(1, n // 5)
    assignment = {}
    for tag in tags[: n - n_val - n_test]:
        assignment[tag] = "train"
    for tag in tags[n - n_val - n_test : n - n_test]:
        assignment[tag] = "val"
    for tag in tags[n - n_test :]:
        assignment[tag] = "test"
    return assignment


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsv", type=Path, default=DEFAULT_TSV)
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be downloaded/extracted without doing it.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = load_rows(args.tsv)
    if not rows:
        sys.exit(
            f"No rows in {args.tsv}. Open the file, follow the header comments, "
            f"and add YouTube URLs for the gap words."
        )

    # Group for reporting.
    per_gloss = defaultdict(list)
    for r in rows:
        per_gloss[r["gloss"]].append(r)

    signer_tags = {r["signer_tag"] for r in rows}
    splits = assign_splits(signer_tags, seed=args.seed)

    print(f"Loaded {len(rows)} row(s) across {len(per_gloss)} gloss(es) "
          f"and {len(signer_tags)} signer(s).")
    print("Signer → split assignment:")
    for tag, split in sorted(splits.items()):
        print(f"  {tag:24s} → {split}")
    print()

    if args.dry_run:
        for gloss, gloss_rows in sorted(per_gloss.items()):
            print(f"[{gloss}] {len(gloss_rows)} row(s)")
            for r in gloss_rows:
                print(f"  {r['signer_tag']:20s} {r['url']}  [{r['start']}-{r['end']}s]"
                      f" → {splits[r['signer_tag']]}")
        print("\n--dry-run: no downloads or extraction performed.")
        return

    OUT_BASE.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        (OUT_BASE / split).mkdir(parents=True, exist_ok=True)

    landmarker = _build_landmarker()
    report = {
        "seq_len": args.seq_len,
        "totals": {"ok": 0, "download_failed": 0, "extract_failed": 0},
        "glosses": {},
    }

    for gloss, gloss_rows in sorted(per_gloss.items()):
        per_class = {"ok": 0, "download_failed": 0, "extract_failed": 0,
                     "splits": {"train": 0, "val": 0, "test": 0}}
        for idx, row in enumerate(gloss_rows):
            split = splits[row["signer_tag"]]
            raw_path = RAW_DIR / gloss / f"{row['signer_tag']}_{idx:03d}.mp4"
            if not download_clip(row, raw_path):
                per_class["download_failed"] += 1
                print(f"  ✗ {gloss}/{row['signer_tag']}#{idx}  download failed")
                continue

            seq = extract_clip(raw_path, landmarker, args.seq_len)
            if seq is None:
                per_class["extract_failed"] += 1
                print(f"  ✗ {gloss}/{row['signer_tag']}#{idx}  no hands detected")
                continue

            sh = signer_hash(row["signer_tag"])
            out_path = OUT_BASE / split / f"{gloss}_syt{sh}_{idx:04d}.npy"
            np.save(str(out_path), seq.astype(np.float32))
            per_class["ok"] += 1
            per_class["splits"][split] += 1
            print(f"  + {gloss}/{row['signer_tag']}#{idx} → {split}/")

        print(f"[{gloss}] ok={per_class['ok']} "
              f"(train={per_class['splits']['train']}, "
              f"val={per_class['splits']['val']}, "
              f"test={per_class['splits']['test']})  "
              f"dl_fail={per_class['download_failed']}  "
              f"extract_fail={per_class['extract_failed']}")
        report["glosses"][gloss] = per_class
        for k in ("ok", "download_failed", "extract_failed"):
            report["totals"][k] += per_class[k]

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"\nReport: {REPORT_PATH}")
    print(f"Totals: {report['totals']}")


if __name__ == "__main__":
    main()
