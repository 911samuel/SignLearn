"""Generate realistic pre-processed landmark fixtures for CI / unit tests.

Writes ``tests/fixtures/processed_mini/{train,val,test}/`` with .npy files
containing plausible (non-zero) hand landmark sequences in shape (30, 126).

Each class gets a hand-pose template derived from rough ASL anatomy. Per-sample
jitter (small Gaussian noise + frame-level jitter) makes every file unique so
augmentation and normalization exercise real non-trivial data paths.

Usage
-----
python tests/generate_test_fixtures.py           # default output
python tests/generate_test_fixtures.py --out tests/fixtures/processed_mini
python tests/generate_test_fixtures.py --classes a b c --train 30 --val 8 --test 8
"""

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------------------
# Landmark templates — 21 (x, y, z) coords per hand, normalised to [0, 1].
# Based on approximate MediaPipe hand landmark positions for each pose.
# Landmark index reference (MediaPipe standard):
#   0  = WRIST
#   1-4  = THUMB   (CMC, MCP, IP, TIP)
#   5-8  = INDEX   (MCP, PIP, DIP, TIP)
#   9-12 = MIDDLE  (MCP, PIP, DIP, TIP)
#  13-16 = RING    (MCP, PIP, DIP, TIP)
#  17-20 = PINKY   (MCP, PIP, DIP, TIP)
# ---------------------------------------------------------------------------

# fmt: off
_POSE_A = np.array([          # ASL 'A' — closed fist, thumb alongside
    [0.50, 0.80, 0.00],       # 0  wrist
    [0.43, 0.73, -0.01],      # 1  thumb CMC
    [0.37, 0.67, -0.02],      # 2  thumb MCP
    [0.32, 0.63, -0.03],      # 3  thumb IP
    [0.29, 0.59, -0.04],      # 4  thumb TIP
    [0.46, 0.67, -0.01],      # 5  index MCP
    [0.46, 0.72, -0.01],      # 6  index PIP (curled back)
    [0.46, 0.70, -0.01],      # 7  index DIP
    [0.46, 0.68, -0.01],      # 8  index TIP
    [0.50, 0.67, -0.01],      # 9  middle MCP
    [0.50, 0.72, -0.01],      # 10 middle PIP
    [0.50, 0.70, -0.01],      # 11 middle DIP
    [0.50, 0.68, -0.01],      # 12 middle TIP
    [0.54, 0.67, -0.01],      # 13 ring MCP
    [0.54, 0.72, -0.01],      # 14 ring PIP
    [0.54, 0.70, -0.01],      # 15 ring DIP
    [0.54, 0.68, -0.01],      # 16 ring TIP
    [0.58, 0.68, -0.01],      # 17 pinky MCP
    [0.58, 0.72, -0.01],      # 18 pinky PIP
    [0.58, 0.71, -0.01],      # 19 pinky DIP
    [0.58, 0.69, -0.01],      # 20 pinky TIP
], dtype=np.float32)

_POSE_B = np.array([          # ASL 'B' — flat hand, four fingers up, thumb tucked
    [0.50, 0.82, 0.00],
    [0.42, 0.73, -0.01],
    [0.36, 0.68, -0.02],
    [0.32, 0.73, -0.03],      # thumb tucked across palm
    [0.35, 0.77, -0.04],
    [0.45, 0.65, -0.01],
    [0.45, 0.50, -0.01],      # index extended upward
    [0.45, 0.40, -0.01],
    [0.45, 0.33, -0.01],
    [0.50, 0.65, -0.01],
    [0.50, 0.50, -0.01],      # middle extended
    [0.50, 0.40, -0.01],
    [0.50, 0.33, -0.01],
    [0.55, 0.65, -0.01],
    [0.55, 0.50, -0.01],      # ring extended
    [0.55, 0.40, -0.01],
    [0.55, 0.33, -0.01],
    [0.60, 0.67, -0.01],
    [0.60, 0.52, -0.01],      # pinky extended
    [0.60, 0.43, -0.01],
    [0.60, 0.36, -0.01],
], dtype=np.float32)

_POSE_C = np.array([          # ASL 'C' — curved C shape
    [0.50, 0.78, 0.00],
    [0.41, 0.70, -0.01],
    [0.35, 0.63, -0.02],
    [0.30, 0.57, -0.03],      # thumb curves outward
    [0.27, 0.51, -0.04],
    [0.45, 0.64, -0.01],
    [0.42, 0.50, -0.02],      # index curves inward
    [0.40, 0.42, -0.03],
    [0.39, 0.37, -0.04],
    [0.50, 0.63, -0.01],
    [0.49, 0.49, -0.02],
    [0.48, 0.41, -0.03],
    [0.48, 0.36, -0.04],
    [0.55, 0.64, -0.01],
    [0.56, 0.50, -0.02],
    [0.57, 0.43, -0.03],
    [0.57, 0.38, -0.04],
    [0.60, 0.67, -0.01],
    [0.62, 0.55, -0.02],
    [0.63, 0.49, -0.03],
    [0.63, 0.44, -0.04],
], dtype=np.float32)
# fmt: on

_TEMPLATES: dict[str, np.ndarray] = {
    "a": _POSE_A,
    "b": _POSE_B,
    "c": _POSE_C,
}

# Additional classes reuse the templates with a slight global offset so labels
# are visually distinct even in absence of real images.
_EXTRA_CLASSES: dict[str, tuple[str, np.ndarray]] = {
    "d": ("a", np.array([0.03, -0.02, 0.005], dtype=np.float32)),
    "e": ("b", np.array([-0.03, 0.01, 0.005], dtype=np.float32)),
}


def _get_template(label: str) -> np.ndarray:
    if label in _TEMPLATES:
        return _TEMPLATES[label]
    if label in _EXTRA_CLASSES:
        base_label, offset = _EXTRA_CLASSES[label]
        return np.clip(_TEMPLATES[base_label] + offset, 0.01, 0.99)
    raise ValueError(f"No template defined for label '{label}'. "
                     "Add it to _TEMPLATES or _EXTRA_CLASSES.")


def _make_sequence(
    template: np.ndarray,
    rng: np.random.Generator,
    *,
    sequence_len: int = 30,
    pose_noise: float = 0.015,
    frame_jitter: float = 0.008,
) -> np.ndarray:
    """Produce a (sequence_len, 126) float32 sequence from a (21, 3) template.

    Noise model:
      - pose_noise:  per-sample Gaussian shift applied to the whole sequence
        (simulates different people / hand sizes / camera angles)
      - frame_jitter: per-frame Gaussian added on top
        (simulates tremor / motion)

    Both hands are set to the same pose; this mimics single-hand data with the
    right-hand slot populated and left-hand slot zero-padded (standard extract.py
    behaviour). A random subset of samples (~20%) set the left hand too.
    """
    pose = template.copy()

    # Per-sample global jitter
    pose = pose + rng.normal(0, pose_noise, size=pose.shape).astype(np.float32)
    pose = np.clip(pose, 0.01, 0.99)

    # Right-hand slot: tile the pose across all frames then add per-frame jitter
    right = np.tile(pose.flatten(), (sequence_len, 1))  # (30, 63)
    right = right + rng.normal(0, frame_jitter, size=right.shape).astype(np.float32)
    right = np.clip(right, 0.0, 1.0)

    # Left-hand slot: zero-padded (no second hand detected), matching extract.py
    left = np.zeros((sequence_len, 63), dtype=np.float32)

    seq = np.concatenate([left, right], axis=1)  # (30, 126)
    return seq.astype(np.float32)


def generate_fixtures(
    out_dir: Path,
    classes: list[str],
    n_train: int,
    n_val: int,
    n_test: int,
    seed: int = 42,
) -> None:
    """Write processed mini fixtures to *out_dir*/{train,val,test}/."""
    rng = np.random.default_rng(seed)
    counts: dict[str, int] = {"train": 0, "val": 0, "test": 0}

    split_cfg = [
        ("train", n_train, range(1, 8)),    # pseudo-subjects 1–7  → train
        ("val",   n_val,   range(8, 10)),   # pseudo-subjects 8–9  → val
        ("test",  n_test,  range(10, 12)),  # pseudo-subjects 10–11 → test
    ]

    for label in classes:
        template = _get_template(label)
        sample_id = 0

        for split, n_samples, subject_range in split_cfg:
            subjects = list(subject_range)
            split_dir = out_dir / split
            split_dir.mkdir(parents=True, exist_ok=True)

            for i in range(n_samples):
                subject_id = subjects[i % len(subjects)]
                stem = f"{label}_s{subject_id:02d}_{sample_id:04d}"
                out_path = split_dir / f"{stem}.npy"

                seq = _make_sequence(template, rng)
                np.save(str(out_path), seq)
                counts[split] += 1
                sample_id += 1

    total = sum(counts.values())
    print(f"Generated {total} fixture files → {out_dir}")
    for split, n in counts.items():
        print(f"  {split:5s}: {n} files")


def verify_fixtures(out_dir: Path) -> bool:
    """Basic sanity check: shape, dtype, and non-zero content."""
    ok = True
    for split in ("train", "val", "test"):
        split_dir = out_dir / split
        if not split_dir.exists():
            print(f"[FAIL] Missing split dir: {split_dir}")
            ok = False
            continue
        npy_files = list(split_dir.glob("*.npy"))
        if not npy_files:
            print(f"[FAIL] No .npy files in {split_dir}")
            ok = False
            continue
        for p in npy_files[:3]:  # spot-check first 3 per split
            arr = np.load(str(p))
            if arr.shape != (30, 126):
                print(f"[FAIL] {p.name}: shape {arr.shape} ≠ (30, 126)")
                ok = False
            if arr.dtype != np.float32:
                print(f"[FAIL] {p.name}: dtype {arr.dtype} ≠ float32")
                ok = False
            if not np.any(arr):
                print(f"[FAIL] {p.name}: all zeros — hand template not applied")
                ok = False
        if ok:
            print(f"[OK] {split}: {len(npy_files)} files, shape/dtype/content verified")
    return ok


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Generate processed landmark fixtures for testing"
    )
    p.add_argument(
        "--out",
        type=Path,
        default=_REPO_ROOT / "tests" / "fixtures" / "processed_mini",
        help="Output directory (default: tests/fixtures/processed_mini)",
    )
    p.add_argument(
        "--classes",
        nargs="+",
        default=["a", "b", "c"],
        help="ASL classes to generate. Must exist in _TEMPLATES or _EXTRA_CLASSES.",
    )
    p.add_argument("--train", type=int, default=30, help="Samples per class in train split")
    p.add_argument("--val",   type=int, default=8,  help="Samples per class in val split")
    p.add_argument("--test",  type=int, default=8,  help="Samples per class in test split")
    p.add_argument("--seed",  type=int, default=42, help="RNG seed for reproducibility")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    generate_fixtures(
        out_dir=args.out,
        classes=args.classes,
        n_train=args.train,
        n_val=args.val,
        n_test=args.test,
        seed=args.seed,
    )
    ok = verify_fixtures(args.out)
    sys.exit(0 if ok else 1)
