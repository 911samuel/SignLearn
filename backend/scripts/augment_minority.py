"""Augment under-represented classes to a target sample count.

Generates synthetic training sequences by applying the full augmentation
pipeline to existing sequences. Only operates on the *train* split and only
on classes below --target-count.

Usage
-----
python backend/scripts/augment_minority.py
python backend/scripts/augment_minority.py --target-count 600 --classes zero one two
python backend/scripts/augment_minority.py --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from backend.data.augment import random_augment, TRAINING_PROBS
from backend.data.constants import FEATURE_DIM, SEQUENCE_LEN
from backend.data.label_map import resolve_label


_DEFAULT_TARGET = 600
_SIGNER_TRAIN_MAX = 7   # signer IDs 01-07 are the train set
_SYNTHETIC_SIGNER = "07"  # tag synthetic samples with signer 07 (stays in train)


def _load_existing(split_dir: Path, label: str) -> list[np.ndarray]:
    """Load all .npy sequences for *label* from *split_dir* (flat layout)."""
    seqs = []
    for f in split_dir.glob(f"{label}_s*.npy"):
        arr = np.load(f)
        if arr.shape == (SEQUENCE_LEN, FEATURE_DIM):
            seqs.append(arr.astype(np.float32))
    return seqs


def _next_index(split_dir: Path, label: str) -> int:
    """Return the next free numeric index for synthetic files."""
    existing = list(split_dir.glob(f"{label}_s{_SYNTHETIC_SIGNER}_*.npy"))
    if not existing:
        return 0
    indices = []
    for f in existing:
        try:
            indices.append(int(f.stem.split("_")[-1]))
        except ValueError:
            pass
    return max(indices) + 1 if indices else 0


def augment_minority(
    train_dir: Path,
    target_count: int,
    classes: list[str] | None,
    dry_run: bool,
    resolve: bool = True,
) -> dict[str, int]:
    """Generate synthetic sequences for under-represented classes.

    Returns a dict mapping label → number of sequences generated.

    Parameters
    ----------
    resolve:
        When True (default) resolve numeric aliases like ``"0"`` → ``"zero"``
        before looking up files. Set to False to operate on raw filename labels.
    """
    generated: dict[str, int] = {}

    # Collect all labels present in the train split
    all_labels = sorted({f.stem.split("_")[0] for f in train_dir.glob("*.npy")})

    if classes:
        if resolve:
            # Deduplicate after resolving (e.g. "0" and "zero" → "zero" once)
            seen = set()
            labels_to_augment = []
            for c in classes:
                resolved = resolve_label(c) or c
                if resolved not in seen:
                    seen.add(resolved)
                    labels_to_augment.append(resolved)
        else:
            labels_to_augment = list(dict.fromkeys(classes))  # preserve order, dedup
    else:
        labels_to_augment = all_labels

    for label in labels_to_augment:
        existing = _load_existing(train_dir, label)
        current_count = len(existing)

        if current_count >= target_count:
            print(f"  {label:15s} {current_count:4d} seqs — already at target, skip")
            continue

        need = target_count - current_count
        print(f"  {label:15s} {current_count:4d} → {target_count} (generate {need})")

        if dry_run or not existing:
            generated[label] = 0
            continue

        next_idx = _next_index(train_dir, label)
        n_gen = 0
        rng_sources = existing.copy()

        while n_gen < need:
            # Pick a random source sequence to augment
            src = rng_sources[n_gen % len(rng_sources)]
            aug = random_augment(src, probs=TRAINING_PROBS)
            fname = train_dir / f"{label}_s{_SYNTHETIC_SIGNER}_{next_idx + n_gen:04d}.npy"
            np.save(fname, aug.astype(np.float32))
            n_gen += 1

        generated[label] = n_gen

    return generated


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Augment under-represented classes to a target count")
    p.add_argument("--train-dir", type=Path, default=Path("data/processed/train"))
    p.add_argument(
        "--target-count", type=int, default=_DEFAULT_TARGET,
        help=f"Minimum samples per class after augmentation (default: {_DEFAULT_TARGET})",
    )
    p.add_argument(
        "--classes", nargs="*", default=None,
        help="Restrict to these class labels (default: all under-target classes)",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be generated without writing files",
    )
    p.add_argument(
        "--no-resolve", action="store_true",
        help="Treat class names literally (do not resolve '0' → 'zero'). "
             "Useful to augment the numeric-label files separately.",
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()

    print(f"Target count: {args.target_count} seqs/class")
    print(f"Train dir: {args.train_dir}")
    if args.dry_run:
        print("DRY RUN — no files will be written")
    print()

    generated = augment_minority(
        train_dir=args.train_dir,
        target_count=args.target_count,
        classes=args.classes,
        dry_run=args.dry_run,
        resolve=not args.no_resolve,
    )

    total = sum(generated.values())
    if not args.dry_run and total:
        print(f"\nGenerated {total} synthetic sequences across {len(generated)} classes.")
        print("Re-run audit to confirm balance: python backend/scripts/audit_dataset.py")
    elif args.dry_run:
        print(f"\n[dry-run] Would generate ~{sum((max(0, args.target_count - len(_load_existing(args.train_dir, lbl))) for lbl in (args.classes or sorted({f.stem.split('_')[0] for f in args.train_dir.glob('*.npy')})))):,} sequences.")
