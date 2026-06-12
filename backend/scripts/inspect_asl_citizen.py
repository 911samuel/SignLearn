"""
Inspect the ASL Citizen dataset zip and report what's inside.

Runs without unzipping the whole archive — uses zipfile to peek at structure,
the gloss vocabulary, and the train/val/test split files. Then writes:

  artifacts/reports/asl_citizen_inventory.md
  configs/asl_citizen_demo_words.txt   (auto-suggested demo vocabulary)

Checks for our priority conversational signs:
  hello, thank_you/THANK_YOU, please, yes, no, help, sorry, goodbye, how, what

Usage:
  python backend/scripts/inspect_asl_citizen.py
  python backend/scripts/inspect_asl_citizen.py --zip data/raw/ASL_Citizen.zip
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
DEFAULT_ZIP = REPO_ROOT / "data" / "raw" / "ASL_Citizen.zip"
OUT_MD = REPO_ROOT / "artifacts" / "reports" / "asl_citizen_inventory.md"
OUT_WORDS = REPO_ROOT / "configs" / "asl_citizen_demo_words.txt"

# Words we want for the conversation demo. Match case-insensitively against
# whatever capitalization ASL Citizen uses (ASL-LEX commonly uses ALL CAPS).
PRIORITY_WORDS = [
    "hello", "hi",
    "thank_you", "thank you", "thanks",
    "please",
    "yes",
    "no",
    "help",
    "sorry",
    "goodbye", "bye",
    "how", "how_are_you",
    "what",
    "name",
    "you",
    "me", "i",
    "good", "bad",
    "fine",
    "nice", "meet",
    "understand",
    "again",
    "where", "when", "why",
]


def _norm(s: str) -> str:
    """Lowercase, strip, replace separators — for fuzzy matching."""
    return re.sub(r"[\s_-]+", "_", s.strip().lower())


def find_csvs(zf: zipfile.ZipFile) -> list[str]:
    return [n for n in zf.namelist() if n.lower().endswith(".csv")]


def find_videos_dir(zf: zipfile.ZipFile) -> str | None:
    for n in zf.namelist():
        if n.lower().endswith(".mp4"):
            # Return the prefix up to the first .mp4.
            return n.rsplit("/", 1)[0] + "/"
    return None


def load_csv(zf: zipfile.ZipFile, name: str) -> list[dict]:
    with zf.open(name) as f:
        text = io.TextIOWrapper(f, encoding="utf-8")
        return list(csv.DictReader(text))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    args = parser.parse_args()

    if not args.zip.exists():
        sys.exit(f"Zip not found: {args.zip}\n"
                 f"Wait for the download to finish, or re-run the curl command.")

    size_mb = args.zip.stat().st_size / 1e6
    print(f"Zip: {args.zip}  ({size_mb:,.0f} MB)")

    with zipfile.ZipFile(args.zip, "r") as zf:
        names = zf.namelist()
        print(f"Entries: {len(names):,}")

        csvs = find_csvs(zf)
        print(f"\nCSV files found ({len(csvs)}):")
        for c in csvs[:20]:
            print(f"  {c}")

        # Look at the first CSV that mentions splits or glosses.
        split_csvs = [c for c in csvs if any(k in c.lower() for k in ("split", "train", "val", "test"))]
        gloss_csvs = [c for c in csvs if any(k in c.lower() for k in ("gloss", "label", "vocab", "sign"))]
        all_csvs = split_csvs or gloss_csvs or csvs

        per_split_counts: dict[str, int] = defaultdict(int)
        per_split_signers: dict[str, set] = defaultdict(set)
        per_gloss_counts: Counter = Counter()
        per_gloss_signers: dict[str, set] = defaultdict(set)
        sample_row = None
        used_csv = None

        for csv_name in all_csvs:
            rows = load_csv(zf, csv_name)
            if not rows:
                continue
            sample_row = rows[0]
            used_csv = csv_name
            # Try to detect column names.
            cols = {k.lower(): k for k in rows[0].keys()}
            gloss_col = next((cols[k] for k in cols if "gloss" in k or "label" in k or "sign" in k), None)
            split_col = next((cols[k] for k in cols if "split" in k), None)
            signer_col = next((cols[k] for k in cols if "signer" in k or "user" in k or "participant" in k), None)
            if gloss_col is None:
                continue
            for r in rows:
                g = r[gloss_col]
                per_gloss_counts[g] += 1
                if split_col:
                    per_split_counts[r[split_col]] += 1
                if signer_col:
                    s = r[signer_col]
                    if split_col:
                        per_split_signers[r[split_col]].add(s)
                    per_gloss_signers[g].add(s)
            print(f"\nUsing CSV: {csv_name}")
            print(f"  Columns: {list(rows[0].keys())}")
            print(f"  Rows: {len(rows):,}")
            break

        videos_dir = find_videos_dir(zf)
        print(f"\nVideo prefix: {videos_dir}")
        n_videos = sum(1 for n in names if n.lower().endswith(".mp4"))
        print(f"Total .mp4 entries: {n_videos:,}")

    # Vocabulary analysis.
    n_glosses = len(per_gloss_counts)
    print(f"\nDistinct glosses: {n_glosses:,}")
    if per_gloss_counts:
        clips_per = list(per_gloss_counts.values())
        clips_per.sort()
        print(f"Clips per gloss — min {clips_per[0]}, "
              f"median {clips_per[len(clips_per)//2]}, "
              f"max {clips_per[-1]}, "
              f"mean {sum(clips_per)/len(clips_per):.1f}")

    if per_split_counts:
        print(f"\nSplit counts:")
        for split, n in per_split_counts.items():
            n_signers = len(per_split_signers.get(split, set()))
            print(f"  {split}: {n:,} clips ({n_signers} signers)")

    # Priority word match.
    norm_glosses = {_norm(g): g for g in per_gloss_counts}
    matches: list[tuple[str, str, int, int]] = []
    for wanted in PRIORITY_WORDS:
        key = _norm(wanted)
        if key in norm_glosses:
            actual = norm_glosses[key]
            matches.append((wanted, actual, per_gloss_counts[actual],
                            len(per_gloss_signers[actual])))

    print(f"\nPriority word matches ({len(matches)} / {len(PRIORITY_WORDS)}):")
    for wanted, actual, n_clips, n_signers in matches:
        print(f"  {wanted:15s} → '{actual}'  {n_clips} clips, {n_signers} signers")

    missing = [w for w in PRIORITY_WORDS if _norm(w) not in norm_glosses]
    if missing:
        print(f"\nPriority words NOT found ({len(missing)}):")
        for w in missing:
            print(f"  {w}")

    # Write report.
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# ASL Citizen Inventory", ""]
    lines.append(f"- Zip size: **{size_mb:,.0f} MB**")
    lines.append(f"- Total entries: {len(names):,}")
    lines.append(f"- Total .mp4 clips: {n_videos:,}")
    lines.append(f"- Distinct glosses: {n_glosses:,}")
    if used_csv:
        lines.append(f"- Manifest CSV: `{used_csv}`")
        lines.append(f"- Sample row: `{sample_row}`")
    lines.append("")
    if per_split_counts:
        lines.append("## Splits")
        lines.append("")
        lines.append("| Split | Clips | Signers |")
        lines.append("|---|---:|---:|")
        for split, n in per_split_counts.items():
            lines.append(f"| {split} | {n:,} | {len(per_split_signers.get(split, set()))} |")
        lines.append("")
    lines.append("## Priority-word coverage")
    lines.append("")
    lines.append("| Wanted | Found gloss | Clips | Signers |")
    lines.append("|---|---|---:|---:|")
    for wanted, actual, n_clips, n_signers in matches:
        lines.append(f"| {wanted} | `{actual}` | {n_clips} | {n_signers} |")
    if missing:
        lines.append("")
        lines.append(f"**Not found in vocabulary:** {', '.join(missing)}")
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {OUT_MD}")

    # Write the demo word list (one gloss per line, in the dataset's own casing).
    OUT_WORDS.parent.mkdir(parents=True, exist_ok=True)
    OUT_WORDS.write_text("\n".join(actual for _, actual, _, _ in matches) + "\n")
    print(f"Wrote {OUT_WORDS}  ({len(matches)} demo glosses)")


if __name__ == "__main__":
    main()
