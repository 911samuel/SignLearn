"""Guided webcam recorder for ASL vocabulary classes.

Displays the word/sign to perform, counts down, captures 30 frames per sample,
saves to data/processed/{split}/, and tracks progress toward the target count.

Usage
-----
# Record hello + thank_you + please (30 samples each, signer 1 → train split)
python backend/scripts/record_vocabulary.py

# Custom word list and signer ID
python backend/scripts/record_vocabulary.py --words hello yes no help --signer 2 --target 50

# Continue an interrupted session (skips completed words)
python backend/scripts/record_vocabulary.py --resume

Controls (OpenCV window must be in focus)
-----------------------------------------
SPACE   Start capturing the current sample
S       Skip to the next word (if this sign is too hard right now)
R       Re-record the last sample (replace it)
Q / ESC Quit and save progress

Signer → split mapping (matches backend/data/extract.py pseudo-subject logic)
  --signer 1-7  → train   (default: 1)
  --signer 8-9  → val
  --signer 10-11→ test
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions

_REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from backend.data.constants import FEATURE_DIM, HAND_DIM, SEQUENCE_LEN
from backend.data.extract import ensure_model

# ---------------------------------------------------------------------------
# Priority word list for smooth deaf communication
# ---------------------------------------------------------------------------

DEFAULT_WORDS = [
    "hello", "thank_you", "please", "yes", "no",
    "help", "stop", "sorry", "good", "bad",
    "eat", "drink", "water", "name", "understand",
]

# Signer ID → split (matches extract.py SPLIT_MAP)
_SIGNER_SPLIT: dict[int, str] = {
    **{i: "train" for i in range(1, 8)},
    8: "val", 9: "val",
    10: "test", 11: "test",
}

# Visual constants
_GREEN  = (0, 220, 0)
_YELLOW = (0, 200, 220)
_RED    = (0, 0, 220)
_WHITE  = (255, 255, 255)
_DARK   = (30, 30, 30)

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _draw_landmarks(frame: np.ndarray, landmarks, w: int, h: int) -> None:
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], _GREEN, 2)
    for pt in pts:
        cv2.circle(frame, pt, 4, _GREEN, -1)


def _overlay(frame: np.ndarray, label: str, sample_idx: int, target: int,
             state: str, countdown: int | None = None) -> None:
    """Draw HUD text onto the frame in-place."""
    h, w = frame.shape[:2]
    # Semi-transparent top bar
    bar = frame.copy()
    cv2.rectangle(bar, (0, 0), (w, 70), _DARK, -1)
    cv2.addWeighted(bar, 0.6, frame, 0.4, 0, frame)

    # Word label
    cv2.putText(frame, f"SIGN:  {label.replace('_', ' ').upper()}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, _WHITE, 2)
    # Progress
    cv2.putText(frame, f"Samples: {sample_idx}/{target}",
                (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.65, _YELLOW, 1)

    if state == "ready":
        msg = "Press SPACE to record  |  S=skip  |  Q=quit"
        cv2.putText(frame, msg, (w // 2 - 230, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, _WHITE, 1)

    elif state == "countdown" and countdown is not None:
        txt = str(countdown) if countdown > 0 else "GO!"
        color = _RED if countdown > 0 else _GREEN
        size = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 4, 6)[0]
        cv2.putText(frame, txt, ((w - size[0]) // 2, (h + size[1]) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 4, color, 6)

    elif state == "recording":
        cv2.circle(frame, (w - 25, 25), 10, _RED, -1)
        cv2.putText(frame, "REC", (w - 65, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, _RED, 2)

    elif state == "saved":
        cv2.putText(frame, "Saved!", (w // 2 - 50, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, _GREEN, 3)


_ANGLE_CHOICES    = ("front", "30L", "30R")
_LIGHTING_CHOICES = ("bright", "dim")


def _write_diversity_sidecar(
    npy_path: Path,
    angle: str,
    lighting: str,
    signer: int,
    label: str,
) -> None:
    """Write a JSON sidecar next to the .npy file with diversity metadata."""
    import json
    meta = {
        "label": label,
        "signer": signer,
        "angle": angle,
        "lighting": lighting,
        "npy": npy_path.name,
    }
    sidecar = npy_path.with_suffix(".json")
    sidecar.write_text(json.dumps(meta, indent=2))


def _prompt_diversity(label: str, sample_idx: int) -> tuple[str, str]:
    """Console prompt for angle + lighting metadata before a take."""
    print(f"\n  [diversity] Take #{sample_idx + 1} for '{label}'")
    print(f"  Angle   : {' / '.join(f'{i}={c}' for i,c in enumerate(_ANGLE_CHOICES))}")
    a_raw = input("  Choose angle [0]: ").strip() or "0"
    try:
        angle = _ANGLE_CHOICES[int(a_raw)]
    except (ValueError, IndexError):
        angle = _ANGLE_CHOICES[0]

    print(f"  Lighting: {' / '.join(f'{i}={c}' for i,c in enumerate(_LIGHTING_CHOICES))}")
    l_raw = input("  Choose lighting [0]: ").strip() or "0"
    try:
        lighting = _LIGHTING_CHOICES[int(l_raw)]
    except (ValueError, IndexError):
        lighting = _LIGHTING_CHOICES[0]

    print(f"  → angle={angle}  lighting={lighting}")
    return angle, lighting


def _diversity_summary(split_dir: Path, label: str) -> dict:
    """Return a dict counting angle × lighting combos recorded so far."""
    import json
    counts: dict[str, dict[str, int]] = {}
    for jf in split_dir.glob(f"{label}_s*.json"):
        try:
            meta = json.loads(jf.read_text())
            a = meta.get("angle", "?")
            l = meta.get("lighting", "?")
            counts.setdefault(a, {}).setdefault(l, 0)
            counts[a][l] += 1
        except Exception:  # noqa: BLE001
            pass
    return counts


def _count_existing(out_dir: Path, label: str) -> int:
    return len(list(out_dir.glob(f"{label}_s*.npy")))


def _next_sample_id(out_dir: Path, label: str) -> int:
    existing = list(out_dir.glob(f"{label}_s*.npy"))
    if not existing:
        return 0
    ids = []
    for p in existing:
        # stem: <label>_s<signer>_<idx>
        parts = p.stem.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            ids.append(int(parts[1]))
    return max(ids) + 1 if ids else len(existing)


# ---------------------------------------------------------------------------
# Core recorder
# ---------------------------------------------------------------------------

def record_session(
    words: list[str],
    out_dir: Path,
    signer: int,
    target: int,
    countdown_secs: int = 3,
    diversity_matrix: bool = False,
) -> dict[str, int]:
    """Run the interactive recording session. Returns {word: n_recorded}."""
    split = _SIGNER_SPLIT.get(signer, "train")
    split_dir = out_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)

    model_path = ensure_model()
    base_options = mp.tasks.BaseOptions(model_asset_path=str(model_path))
    options = HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam. Check camera permissions.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    stats: dict[str, int] = {}
    word_idx = 0

    print(f"\nSignLearn Vocabulary Recorder")
    print(f"  Signer: {signer} → split: {split}")
    print(f"  Target: {target} samples per word")
    print(f"  Words:  {', '.join(words)}")
    if diversity_matrix:
        print(f"  Mode:   diversity-matrix (angle × lighting metadata per take)")
    print("\nControls: SPACE=record  S=skip word  R=redo last  Q=quit\n")

    with HandLandmarker.create_from_options(options) as landmarker:
        start_ms = int(time.time() * 1000)

        while word_idx < len(words):
            label = words[word_idx]
            collected = _count_existing(split_dir, label)

            if collected >= target:
                print(f"  [{label}] Already at target ({collected}/{target}) — skipping.")
                stats[label] = collected
                word_idx += 1
                continue

            print(f"\n--- {label.upper().replace('_', ' ')} ---  "
                  f"({collected}/{target} samples in {split_dir})")

            state = "ready"
            frames: list[np.ndarray] = []
            last_saved_path: Path | None = None
            countdown_start: float | None = None
            countdown_val = countdown_secs
            _pending_angle: str = _ANGLE_CHOICES[0]
            _pending_lighting: str = _LIGHTING_CHOICES[0]

            if diversity_matrix:
                # Show existing coverage before starting the word
                coverage = _diversity_summary(split_dir, label)
                if coverage:
                    print(f"  Current coverage for '{label}':")
                    for angle, lmap in sorted(coverage.items()):
                        for light, cnt in sorted(lmap.items()):
                            print(f"    angle={angle}, lighting={light}: {cnt} takes")

            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Camera read failed.")
                    break

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                ts_ms = int(time.time() * 1000) - start_ms
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect_for_video(mp_image, ts_ms)

                left  = np.zeros(HAND_DIM, dtype=np.float32)
                right = np.zeros(HAND_DIM, dtype=np.float32)
                for lms, handedness_list in zip(result.hand_landmarks, result.handedness):
                    side = handedness_list[0].category_name.lower()
                    row = np.array([[l.x, l.y, l.z] for l in lms], np.float32).flatten()
                    if side == "left":
                        left = row
                    else:
                        right = row
                    _draw_landmarks(frame, lms, w, h)

                frame_data = np.concatenate([left, right]).astype(np.float32)

                # State machine
                if state == "countdown":
                    elapsed = time.time() - countdown_start
                    countdown_val = countdown_secs - int(elapsed)
                    if countdown_val <= 0:
                        state = "recording"
                        frames = []
                        print(f"  Recording...", end="", flush=True)

                elif state == "recording":
                    frames.append(frame_data)
                    if len(frames) >= SEQUENCE_LEN:
                        seq = np.stack(frames[:SEQUENCE_LEN], axis=0).astype(np.float32)
                        sample_id = _next_sample_id(split_dir, label)
                        stem = f"{label}_s{signer:02d}_{sample_id:04d}"
                        out_path = split_dir / f"{stem}.npy"
                        np.save(str(out_path), seq)
                        last_saved_path = out_path
                        if diversity_matrix:
                            _write_diversity_sidecar(
                                out_path, _pending_angle, _pending_lighting,
                                signer, label,
                            )
                        collected += 1
                        print(f" saved ({collected}/{target})")
                        state = "saved"
                        save_start = time.time()
                        stats[label] = collected

                elif state == "saved":
                    if time.time() - save_start > 0.8:
                        if collected >= target:
                            print(f"  [{label}] Target reached!")
                            word_idx += 1
                            break
                        state = "ready"

                # Draw overlay
                cd = countdown_val if state == "countdown" else None
                _overlay(frame, label, collected, target, state, cd)
                cv2.imshow("SignLearn — Vocabulary Recorder", frame)

                key = cv2.waitKey(1) & 0xFF

                if key in (ord("q"), 27):  # Q or ESC
                    print("\nSession ended by user.")
                    cap.release()
                    cv2.destroyAllWindows()
                    return stats

                elif key == ord(" ") and state == "ready":
                    if diversity_matrix:
                        # Pause video loop to collect metadata from console
                        cv2.destroyAllWindows()
                        _pending_angle, _pending_lighting = _prompt_diversity(label, collected)
                        # Reopen window
                        cv2.namedWindow("SignLearn — Vocabulary Recorder")
                    state = "countdown"
                    countdown_start = time.time()
                    countdown_val = countdown_secs

                elif key == ord("s") and state in ("ready", "saved"):
                    print(f"  [{label}] Skipped.")
                    stats[label] = collected
                    word_idx += 1
                    break

                elif key == ord("r") and state == "saved" and last_saved_path is not None:
                    last_saved_path.unlink(missing_ok=True)
                    # Also remove sidecar if present
                    sidecar = last_saved_path.with_suffix(".json")
                    sidecar.unlink(missing_ok=True)
                    collected -= 1
                    stats[label] = collected
                    print(f"  Deleted last sample. ({collected}/{target})")
                    state = "ready"

    cap.release()
    cv2.destroyAllWindows()
    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Guided webcam recorder for ASL vocabulary classes"
    )
    p.add_argument(
        "--words", nargs="+", default=None,
        help="Words to record (default: top-15 communication priority list)",
    )
    p.add_argument(
        "--signer", type=int, default=1, choices=range(1, 12), metavar="1-11",
        help="Signer ID 1-7=train, 8-9=val, 10-11=test (default: 1)",
    )
    p.add_argument(
        "--target", type=int, default=30,
        help="Target samples per word class (default: 30)",
    )
    p.add_argument(
        "--countdown", type=int, default=3,
        help="Countdown seconds before each recording starts (default: 3)",
    )
    p.add_argument(
        "--out", type=Path,
        default=_REPO_ROOT / "data" / "processed",
        help="Processed data root (default: data/processed)",
    )
    p.add_argument(
        "--resume", action="store_true",
        help="Skip words that already have >= target samples",
    )
    p.add_argument(
        "--diversity-matrix", action="store_true",
        help="Prompt for angle (front/30L/30R) and lighting (bright/dim) metadata "
             "before each take; writes a JSON sidecar alongside each .npy file.",
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    words = args.words or DEFAULT_WORDS

    # Quick sanity check: verify all requested words are in the vocabulary
    from backend.data.label_map import load_label_map
    vocab = load_label_map()
    unknown = [w for w in words if w not in vocab]
    if unknown:
        print(f"Unknown vocabulary words: {unknown}")
        print(f"Check docs/vocabulary.md for valid labels.")
        sys.exit(1)

    results = record_session(
        words=words,
        out_dir=args.out,
        signer=args.signer,
        target=args.target,
        diversity_matrix=args.diversity_matrix,
        countdown_secs=args.countdown,
    )

    print("\n--- Session summary ---")
    for word, count in results.items():
        bar = "█" * count + "░" * max(0, args.target - count)
        status = "✓" if count >= args.target else f"{count}/{args.target}"
        print(f"  {word:<15} {bar[:30]}  {status}")

    total = sum(results.values())
    print(f"\nTotal sequences recorded this session: {total}")
    print(f"Run audit:   python backend/scripts/audit_dataset.py")
    print(f"Retrain:     python backend/scripts/train_model.py --arch transformer --run-name tx-v3-words --epochs 100")
