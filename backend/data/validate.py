"""Subtask 8: dataset validation CLI — leakage, distribution, shape, statistical checks.

Exit code 0 = all checks passed.
Exit code 1 = one or more checks failed (CI-friendly).

Usage:
    python -m backend.data.validate --processed data/processed
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from backend.data.label_map import load_label_map, resolve_label

_REPO_ROOT   = Path(__file__).parent.parent.parent
_ARTIFACTS   = _REPO_ROOT / "artifacts"
_STEM_RE     = re.compile(r"^(.+)_s(\d{2})_(\d{4})$")
SEQUENCE_LEN = 30
TWO_HAND_DIM = 126
SPLITS       = ("train", "val", "test")


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_stem(stem: str) -> tuple[str, int, int] | None:
    """Return (vocab_label, subject_id, sample_id) or None on parse failure."""
    m = _STEM_RE.match(stem)
    if not m:
        return None
    raw_label  = m.group(1)
    subject_id = int(m.group(2))
    sample_id  = int(m.group(3))
    return resolve_label(raw_label), subject_id, sample_id


def _scan(processed_dir: Path) -> dict[str, list[tuple[Path, str, int]]]:
    """Return {split: [(path, vocab_label, subject_id), ...]} for all splits."""
    data: dict[str, list] = {s: [] for s in SPLITS}
    for split in SPLITS:
        split_dir = processed_dir / split
        if not split_dir.exists():
            continue
        for npy in sorted(split_dir.glob("*.npy")):
            parsed = _parse_stem(npy.stem)
            if parsed is None:
                continue
            vocab_label, subject_id, _ = parsed
            data[split].append((npy, vocab_label, subject_id))
    return data


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_subject_leakage(
    data: dict[str, list],
) -> list[str]:
    """Return error messages for any subject_id that appears in more than one split."""
    subject_splits: dict[int, set[str]] = defaultdict(set)
    for split, entries in data.items():
        for _, _, subject_id in entries:
            subject_splits[subject_id].add(split)

    errors = []
    for subject_id, splits in sorted(subject_splits.items()):
        if len(splits) > 1:
            errors.append(
                f"LEAKAGE: subject s{subject_id:02d} appears in splits: {sorted(splits)}"
            )
    return errors


def check_shapes_and_integrity(
    data: dict[str, list],
) -> tuple[list[str], int]:
    """Check shape, dtype, NaN for every .npy.

    Returns (error_messages, total_checked).
    """
    errors: list[str] = []
    total = 0
    for split, entries in data.items():
        for path, _, _ in entries:
            total += 1
            try:
                arr = np.load(str(path))
            except Exception as e:
                errors.append(f"LOAD_ERROR: {path}: {e}")
                continue
            if arr.shape != (SEQUENCE_LEN, TWO_HAND_DIM):
                errors.append(
                    f"SHAPE: {path.name} expected ({SEQUENCE_LEN},{TWO_HAND_DIM}), "
                    f"got {arr.shape}"
                )
            if arr.dtype != np.float32:
                errors.append(f"DTYPE: {path.name} expected float32, got {arr.dtype}")
            if np.isnan(arr).any():
                errors.append(f"NAN: {path.name} contains NaN values")
    return errors, total


def build_class_distribution(
    data: dict[str, list],
) -> dict[str, dict[str, int]]:
    """Return {split: {vocab_label: count}} for all splits."""
    dist: dict[str, dict[str, int]] = {}
    for split, entries in data.items():
        counts: dict[str, int] = defaultdict(int)
        for _, label, _ in entries:
            counts[label] += 1
        dist[split] = dict(sorted(counts.items()))
    return dist


def compute_feature_stats(
    data: dict[str, list],
) -> dict[str, list[float]]:
    """Compute per-feature mean and std across the entire train split.

    Returns {"mean": [...126...], "std": [...126...]}.
    """
    train_entries = data.get("train", [])
    if not train_entries:
        return {}

    all_frames: list[np.ndarray] = []
    for path, _, _ in train_entries:
        try:
            arr = np.load(str(path))   # (30, 126)
            all_frames.append(arr)
        except Exception:
            continue

    if not all_frames:
        return {}

    stacked = np.concatenate(all_frames, axis=0)   # (N*30, 126)
    return {
        "mean": stacked.mean(axis=0).tolist(),
        "std":  stacked.std(axis=0).tolist(),
        "n_frames": int(stacked.shape[0]),
    }


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------

def _distribution_table(dist: dict[str, dict[str, int]]) -> str:
    """Return a Markdown table of class counts per split."""
    label_map = load_label_map()
    all_labels = sorted(
        {lbl for counts in dist.values() for lbl in counts},
        key=lambda l: label_map.get(l, 9999),
    )
    header  = "| Class | " + " | ".join(SPLITS) + " |"
    divider = "|-------|" + "---|".join(["---"] * len(SPLITS)) + "|"
    rows    = [header, divider]
    for lbl in all_labels:
        cells = " | ".join(str(dist[s].get(lbl, 0)) for s in SPLITS)
        rows.append(f"| {lbl} | {cells} |")
    totals = " | ".join(str(sum(dist[s].values())) for s in SPLITS)
    rows.append(f"| **TOTAL** | {totals} |")
    return "\n".join(rows)


def write_report(
    errors: list[str],
    dist: dict[str, dict[str, int]],
    total_files: int,
    stats: dict,
    out_path: Path,
) -> None:
    lines = ["# SignLearn Dataset Validation Report\n"]

    status = "PASS" if not errors else "FAIL"
    lines.append(f"**Status**: {status}  \n**Files checked**: {total_files}\n")

    lines.append("## Class Distribution\n")
    lines.append(_distribution_table(dist))
    lines.append("")

    if errors:
        lines.append("## Errors\n")
        for e in errors:
            lines.append(f"- {e}")
        lines.append("")
    else:
        lines.append("## Errors\n\n*None — all checks passed.*\n")

    if stats:
        lines.append("## Feature Statistics (train)\n")
        lines.append(f"- Frames: {stats.get('n_frames', '?')}")
        mean_arr = np.array(stats["mean"])
        std_arr  = np.array(stats["std"])
        lines.append(f"- Mean range: [{mean_arr.min():.4f}, {mean_arr.max():.4f}]")
        lines.append(f"- Std  range: [{std_arr.min():.4f},  {std_arr.max():.4f}]")
        lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    print(f"Wrote report → {out_path}")


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------

def validate(processed_dir: Path, artifacts_dir: Path = _ARTIFACTS) -> list[str]:
    """Run all checks and return a list of error messages (empty = pass)."""
    data = _scan(processed_dir)

    all_errors: list[str] = []

    # 1. Subject leakage
    leakage = check_subject_leakage(data)
    all_errors.extend(leakage)

    # 2. Shape / integrity
    shape_errors, total_files = check_shapes_and_integrity(data)
    all_errors.extend(shape_errors)

    # 3. Distribution
    dist = build_class_distribution(data)

    # 4. Feature stats
    stats = compute_feature_stats(data)
    if stats:
        stats_path = artifacts_dir / "feature_stats.json"
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"Wrote stats  → {stats_path}")

    # 5. Write report
    report_path = artifacts_dir / "validation_report.md"
    write_report(all_errors, dist, total_files, stats, report_path)

    # 6. Console summary
    for split in SPLITS:
        n = sum(dist.get(split, {}).values())
        classes = len(dist.get(split, {}))
        print(f"  {split:5s}: {n:5d} files, {classes} classes")

    if all_errors:
        print(f"\n{'─'*60}")
        print(f"FAILED — {len(all_errors)} error(s):")
        for e in all_errors[:20]:
            print(f"  {e}")
        if len(all_errors) > 20:
            print(f"  … and {len(all_errors) - 20} more (see report)")
    else:
        print(f"\nAll checks PASSED ({total_files} files).")

    return all_errors


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate the processed dataset")
    parser.add_argument(
        "--processed", default="data/processed",
        help="Root of processed splits (default: data/processed)"
    )
    parser.add_argument(
        "--artifacts", default=str(_ARTIFACTS),
        help="Directory for report output (default: artifacts/)"
    )
    args = parser.parse_args()

    errors = validate(
        processed_dir=Path(args.processed),
        artifacts_dir=Path(args.artifacts),
    )
    sys.exit(1 if errors else 0)
