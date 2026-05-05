"""Subtask 2-3: pseudo-subject assignment + MediaPipe batch extraction to (30, 126) .npy."""

import argparse
import hashlib
import logging
import multiprocessing as mp
from pathlib import Path

import cv2
import mediapipe as mp_lib
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEQUENCE_LENGTH = 30
LANDMARK_DIM    = 63    # 21 landmarks × 3 coords per hand
TWO_HAND_DIM    = 126   # left(63) + right(63)

_REPO_ROOT  = Path(__file__).parent.parent.parent
MODEL_PATH  = _REPO_ROOT / "models" / "hand_landmarker.task"
_ARTIFACTS  = _REPO_ROOT / "artifacts"

# ---------------------------------------------------------------------------
# Subtask 2 — pseudo-subject assignment & canonical naming
# ---------------------------------------------------------------------------

_SPLIT_MAP = {
    1: "train", 2: "train", 3: "train", 4: "train",
    5: "train", 6: "train", 7: "train",
    8: "val",   9: "val",
    10: "test", 11: "test",
}


def assign_subject(filename: str, n_subjects: int = 11) -> int:
    """Map a raw image filename to a stable pseudo-subject id in [1, n_subjects].

    Uses MD5 of the file stem so the same filename always yields the same id.
    With n_subjects=11 this enables the 7/2/2 person-independent split.
    """
    stem = filename.rsplit(".", 1)[0]
    digest = hashlib.md5(stem.encode()).hexdigest()
    return int(digest, 16) % n_subjects + 1


def subject_to_split(subject_id: int) -> str:
    """Return 'train', 'val', or 'test' for a given subject_id."""
    return _SPLIT_MAP[subject_id]


def canonical_name(label: str, subject_id: int, sample_id: int) -> str:
    """Return a standardised filename stem: <label>_s<subject_id>_<sample_id>.

    Example: hello_s03_0042
    """
    return f"{label}_s{subject_id:02d}_{sample_id:04d}"


# ---------------------------------------------------------------------------
# Subtask 3 — MediaPipe landmark extraction
# ---------------------------------------------------------------------------

def _build_landmarker() -> HandLandmarker:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"MediaPipe model not found: {MODEL_PATH}\n"
            "Run: curl -L https://storage.googleapis.com/mediapipe-models/"
            "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
            " -o models/hand_landmarker.task"
        )
    base_options = mp_lib.tasks.BaseOptions(model_asset_path=str(MODEL_PATH))
    options = HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
    )
    return HandLandmarker.create_from_options(options)


def _extract_row(landmarks) -> np.ndarray:
    """Flatten 21 MediaPipe landmarks to a (63,) float32 array."""
    return np.array(
        [[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32
    ).flatten()


def extract_two_hands(image_path: Path, landmarker: HandLandmarker) -> np.ndarray:
    """Run MediaPipe on one image; return (126,) float32 [left(63) | right(63)].

    Missing hands are zero-padded. Handedness is taken directly from MediaPipe.
    """
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise ValueError(f"Cannot read image: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp_lib.Image(image_format=mp_lib.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)

    left  = np.zeros(LANDMARK_DIM, dtype=np.float32)
    right = np.zeros(LANDMARK_DIM, dtype=np.float32)

    for landmarks, handedness_list in zip(
        result.hand_landmarks, result.handedness
    ):
        label = handedness_list[0].category_name.lower()  # "left" or "right"
        row = _extract_row(landmarks)
        if label == "left":
            left = row
        else:
            right = row

    return np.concatenate([left, right], dtype=np.float32)


def to_sequence(frame: np.ndarray, target_len: int = SEQUENCE_LENGTH) -> np.ndarray:
    """Replicate a single (126,) landmark frame into a (target_len, 126) sequence.

    For static image data this creates a temporally-uniform tensor that the LSTM
    can treat as a held pose. Augmentation (Subtask 6) adds per-frame jitter.
    """
    assert frame.shape == (TWO_HAND_DIM,), f"Expected ({TWO_HAND_DIM},), got {frame.shape}"
    return np.tile(frame, (target_len, 1)).astype(np.float32)


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def _process_one(args: tuple) -> str | None:
    """Worker: extract one image and write .npy. Returns failure message or None."""
    image_path, out_path, label, subject_id, sample_id = args
    try:
        landmarker = _build_landmarker()
        frame = extract_two_hands(image_path, landmarker)
        seq = to_sequence(frame)

        # Detect total-zero case (no hand at all)
        if not np.any(frame):
            return f"NO_HAND:{image_path}"

        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(out_path), seq)
        return None
    except Exception as e:
        return f"ERROR:{image_path}:{e}"


def process_dataset(
    raw_dir: Path,
    out_dir: Path,
    class_filter: list[str] | None = None,
    workers: int = 4,
) -> None:
    """Batch-extract all images in raw_dir/<class>/*.jpg into out_dir/<split>/.

    Args:
        raw_dir:      e.g. data/raw/digits  (contains subdirs per class)
        out_dir:      e.g. data/processed
        class_filter: optional list of class subdir names to restrict processing
        workers:      number of parallel workers
    """
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda x, **kw: x  # noqa: E731

    _ARTIFACTS.mkdir(parents=True, exist_ok=True)
    failure_log = _ARTIFACTS / "extract_failures.log"

    class_dirs = sorted(d for d in raw_dir.iterdir() if d.is_dir())
    if class_filter:
        class_dirs = [d for d in class_dirs if d.name in class_filter]

    tasks: list[tuple] = []

    for cls_dir in class_dirs:
        label = cls_dir.name
        images = sorted(cls_dir.glob("*.[Jj][Pp][Gg]")) + \
                 sorted(cls_dir.glob("*.[Pp][Nn][Gg]"))

        for sample_id, img_path in enumerate(images):
            subject_id = assign_subject(img_path.name)
            split = subject_to_split(subject_id)
            stem = canonical_name(label, subject_id, sample_id)
            out_path = out_dir / split / f"{stem}.npy"

            if out_path.exists():
                continue  # idempotent — skip already-processed files

            tasks.append((img_path, out_path, label, subject_id, sample_id))

    if not tasks:
        print("All files already processed — nothing to do.")
        return

    print(f"Processing {len(tasks)} images with {workers} workers …")
    failures: list[str] = []

    with mp.Pool(workers) as pool:
        for result in tqdm(pool.imap_unordered(_process_one, tasks), total=len(tasks)):
            if result is not None:
                failures.append(result)

    if failures:
        with open(failure_log, "a") as f:
            f.write("\n".join(failures) + "\n")
        print(f"  {len(failures)} failures logged to {failure_log}")

    total = len(tasks) - len(failures)
    print(f"Done — {total} sequences written to {out_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch landmark extraction → (30,126) .npy")
    parser.add_argument("--raw",     required=True,  help="Raw image directory (e.g. data/raw/digits)")
    parser.add_argument("--out",     default="data/processed", help="Output root directory")
    parser.add_argument("--classes", nargs="*",      help="Restrict to these class subdirs")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers")
    args = parser.parse_args()

    process_dataset(
        raw_dir=Path(args.raw),
        out_dir=Path(args.out),
        class_filter=args.classes,
        workers=args.workers,
    )
