"""Preview or execute the 70/15/15 person-independent split for a raw dataset.

Uses the same MD5-based pseudo-subject assignment as backend/data/extract.py so
results exactly match what batch extraction would produce — no separate split
step is needed before calling extract.py.

Modes
-----
--dry-run  (default)
    Scan *raw-dir*, print a split preview table, and write a JSON report.
    No files are moved or copied. Use this to verify the split before running
    a long extraction job.

--copy  or  --symlink
    Actually copy / symlink raw images into ``<out-dir>/{train,val,test}/<class>/``
    so downstream tools that expect pre-split directories can consume them directly.

Usage
-----
python scripts/split_dataset.py --raw data/raw/alphabet          # dry-run
python scripts/split_dataset.py --raw data/raw/alphabet --copy   # copy files
python scripts/split_dataset.py --raw data/raw/alphabet --symlink --out /tmp/split
"""

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from backend.data.extract import assign_subject, subject_to_split

_IMAGE_GLOBS = ("*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.png", "*.PNG")


def _scan_raw(raw_dir: Path) -> list[tuple[Path, str, str]]:
    """Return [(image_path, class_label, split), ...] for every image in raw_dir."""
    results: list[tuple[Path, str, str]] = []
    class_dirs = sorted(d for d in raw_dir.iterdir() if d.is_dir())
    if not class_dirs:
        raise ValueError(f"No class subdirectories found in {raw_dir}")
    for cls_dir in class_dirs:
        label = cls_dir.name
        images: list[Path] = []
        for glob in _IMAGE_GLOBS:
            images.extend(cls_dir.glob(glob))
        images = sorted(set(images))
        for img in images:
            subject_id = assign_subject(img.name)
            split = subject_to_split(subject_id)
            results.append((img, label, split))
    return results


def _build_distribution(
    items: list[tuple[Path, str, str]]
) -> dict[str, dict[str, int]]:
    """Return {split: {class_label: count}}."""
    dist: dict[str, dict[str, int]] = {
        "train": defaultdict(int),
        "val":   defaultdict(int),
        "test":  defaultdict(int),
    }
    for _, label, split in items:
        dist[split][label] += 1
    return {s: dict(sorted(d.items())) for s, d in dist.items()}


def print_table(dist: dict[str, dict[str, int]], raw_dir: Path) -> None:
    splits = ["train", "val", "test"]
    all_labels = sorted(
        {lbl for d in dist.values() for lbl in d}
    )
    total_images = sum(sum(d.values()) for d in dist.values())

    print(f"\n{'─' * 60}")
    print(f"  Split preview for: {raw_dir}")
    print(f"  Classes: {len(all_labels)}    Images: {total_images}")
    print(f"{'─' * 60}")

    col_w = max(len(l) for l in all_labels) + 2
    header = f"  {'Class':<{col_w}}" + "".join(f"{'  ' + s:>10}" for s in splits) + "   Total"
    print(header)
    print("  " + "─" * (len(header) - 2))

    for lbl in all_labels:
        counts = [dist[s].get(lbl, 0) for s in splits]
        row = f"  {lbl:<{col_w}}" + "".join(f"{c:>10}" for c in counts)
        row += f"   {sum(counts):>5}"
        print(row)

    print("  " + "─" * (len(header) - 2))
    totals = [sum(dist[s].values()) for s in splits]
    grand  = sum(totals)
    print(f"  {'TOTAL':<{col_w}}" + "".join(f"{t:>10}" for t in totals) + f"   {grand:>5}")

    pct = [f"{100 * t / grand:.1f}%" if grand else "—" for t in totals]
    print(f"  {'%':<{col_w}}" + "".join(f"{p:>10}" for p in pct))
    print(f"{'─' * 60}\n")


def write_json_report(
    dist: dict[str, dict[str, int]],
    raw_dir: Path,
    out_path: Path,
) -> None:
    report = {
        "raw_dir": str(raw_dir),
        "distribution": dist,
        "totals": {s: sum(d.values()) for s, d in dist.items()},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Split report → {out_path}")


def execute_split(
    items: list[tuple[Path, str, str]],
    out_dir: Path,
    mode: str,  # "copy" or "symlink"
) -> None:
    """Copy or symlink images into out_dir/{split}/{class}/."""
    counts: dict[str, int] = defaultdict(int)
    for img_path, label, split in items:
        dest_dir = out_dir / split / label
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / img_path.name
        if dest.exists():
            continue
        if mode == "symlink":
            dest.symlink_to(img_path.resolve())
        else:
            shutil.copy2(img_path, dest)
        counts[split] += 1

    total = sum(counts.values())
    print(f"{'Copied' if mode == 'copy' else 'Linked'} {total} files → {out_dir}")
    for split in ("train", "val", "test"):
        print(f"  {split:5s}: {counts.get(split, 0)}")


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Preview or execute 70/15/15 split for a raw ASL image directory"
    )
    p.add_argument(
        "--raw",
        required=True,
        type=Path,
        help="Raw image root (contains subdirs per class, e.g. data/raw/alphabet)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output root for --copy / --symlink (default: <raw>/../split/)",
    )
    p.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Write JSON split report to this path (default: artifacts/split_report.json)",
    )

    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="[default] Print a split preview table without moving files",
    )
    mode.add_argument(
        "--copy",
        action="store_true",
        help="Copy raw images into out_dir/{train,val,test}/<class>/",
    )
    mode.add_argument(
        "--symlink",
        action="store_true",
        help="Create symlinks instead of copying (faster, saves disk space)",
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()

    if not args.raw.exists():
        print(f"Error: raw directory not found: {args.raw}", file=sys.stderr)
        sys.exit(1)

    print(f"Scanning {args.raw} …")
    items = _scan_raw(args.raw)
    dist  = _build_distribution(items)
    print_table(dist, args.raw)

    report_path = args.report or (_REPO_ROOT / "artifacts" / "split_report.json")
    write_json_report(dist, args.raw, report_path)

    if args.copy or args.symlink:
        out_dir = args.out or (args.raw.parent / "split")
        mode = "copy" if args.copy else "symlink"
        execute_split(items, out_dir, mode)
    else:
        print("Dry-run mode — no files moved. Use --copy or --symlink to execute.")
