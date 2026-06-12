"""
Download WLASL word-level ASL clips for the word-recognition model.

WLASL ships only metadata (``WLASL_v0.3.json``); the videos themselves live on
YouTube and other hosts and must be fetched per-entry. Many source URLs are
dead — typical real-world yield is ~60-70%. That is expected; the companion
``audit_wlasl.py`` script reports the actual yield and helps you pick the final
class list.

Prerequisites
-------------
1. Metadata file: download ``WLASL_v0.3.json`` from
   https://github.com/dxli94/WLASL/blob/master/start_kit/WLASL_v0.3.json
   and place it at ``data/raw/wlasl/WLASL_v0.3.json``.

2. yt-dlp:
     pip install yt-dlp
   (We use yt-dlp rather than youtube-dl — it is actively maintained and
   handles the rate-limiting / signature changes that break the original
   WLASL download script.)

3. ffmpeg available on PATH (yt-dlp uses it to trim clips by frame range).

Usage
-----
  # Inspect WLASL-100 word list without downloading
  python backend/scripts/download_wlasl.py --top-k 100 --dry-run

  # Download the top-100 most-frequent glosses
  python backend/scripts/download_wlasl.py --top-k 100

  # Download a curated word list (one gloss per line)
  python backend/scripts/download_wlasl.py --words-file configs/wlasl_words.txt

  # Resume / re-run is safe — already-downloaded clips are skipped.

Layout produced
---------------
  data/raw/wlasl/
    WLASL_v0.3.json
    videos/<gloss>/<video_id>_<inst_idx>.mp4
    download_log.jsonl           # one line per attempted instance
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).parent.parent.parent
WLASL_ROOT = REPO_ROOT / "data" / "raw" / "wlasl"
METADATA_PATH = WLASL_ROOT / "WLASL_v0.3.json"
VIDEOS_DIR = WLASL_ROOT / "videos"
LOG_PATH = WLASL_ROOT / "download_log.jsonl"


def _require_metadata() -> list[dict]:
    if not METADATA_PATH.exists():
        sys.exit(
            f"Missing {METADATA_PATH}.\n"
            "Download WLASL_v0.3.json from "
            "https://github.com/dxli94/WLASL/blob/master/start_kit/WLASL_v0.3.json "
            f"and place it at {METADATA_PATH}."
        )
    with METADATA_PATH.open() as f:
        return json.load(f)


def _require_yt_dlp() -> str:
    binary = shutil.which("yt-dlp")
    if binary:
        return binary
    # Fall back to module invocation if the CLI shim isn't on PATH.
    try:
        import yt_dlp  # noqa: F401
        return f"{sys.executable} -m yt_dlp"
    except ImportError:
        sys.exit("yt-dlp not installed. Run: pip install yt-dlp")


def _select_glosses(meta: list[dict], top_k: int | None, words_file: Path | None) -> list[dict]:
    if words_file is not None:
        wanted = {w.strip().lower() for w in words_file.read_text().splitlines() if w.strip()}
        return [entry for entry in meta if entry["gloss"].lower() in wanted]
    if top_k is not None:
        # WLASL is already ordered by frequency in the canonical release, but
        # sort by instance count defensively in case a custom JSON is used.
        ordered = sorted(meta, key=lambda e: len(e["instances"]), reverse=True)
        return ordered[:top_k]
    return meta


def _append_log(record: dict) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a") as f:
        f.write(json.dumps(record) + "\n")


def _download_instance(
    yt_dlp_cmd: str,
    gloss: str,
    instance: dict,
    inst_idx: int,
) -> dict:
    """Download a single instance. Returns a structured log record."""
    video_id = instance.get("video_id", f"unknown_{inst_idx}")
    url = instance.get("url")
    frame_start = instance.get("frame_start", 1)
    frame_end = instance.get("frame_end", -1)
    signer_id = instance.get("signer_id")
    split = instance.get("split")

    out_dir = VIDEOS_DIR / gloss
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{video_id}_{inst_idx:03d}.mp4"

    base_record = {
        "gloss": gloss,
        "video_id": video_id,
        "inst_idx": inst_idx,
        "signer_id": signer_id,
        "split": split,
        "url": url,
        "frame_start": frame_start,
        "frame_end": frame_end,
    }

    if out_path.exists() and out_path.stat().st_size > 0:
        return {**base_record, "status": "skipped_existing", "path": str(out_path)}

    if not url:
        return {**base_record, "status": "no_url"}

    cmd = (
        yt_dlp_cmd.split()
        + [
            "--quiet",
            "--no-warnings",
            "--no-playlist",
            "-f",
            "mp4/best",
            "-o",
            str(out_path),
            url,
        ]
    )

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return {**base_record, "status": "timeout"}

    if result.returncode != 0 or not out_path.exists():
        return {
            **base_record,
            "status": "download_failed",
            "stderr": (result.stderr or "").strip()[-200:],
        }

    return {**base_record, "status": "ok", "path": str(out_path)}


def download_all(
    selected: list[dict],
    yt_dlp_cmd: str,
    limit_per_class: int | None = None,
) -> Counter:
    status_counts: Counter = Counter()
    for entry in selected:
        gloss = entry["gloss"]
        instances = entry["instances"]
        if limit_per_class is not None:
            instances = instances[:limit_per_class]
        print(f"[{gloss}] {len(instances)} instance(s)")
        for idx, inst in enumerate(instances):
            record = _download_instance(yt_dlp_cmd, gloss, inst, idx)
            _append_log(record)
            status_counts[record["status"]] += 1
            marker = {"ok": "+", "skipped_existing": ".", "no_url": "x"}.get(
                record["status"], "!"
            )
            print(f"  {marker} {gloss}/{inst.get('video_id')}#{idx}  [{record['status']}]")
    return status_counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Download WLASL word-level ASL videos.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--top-k",
        type=int,
        help="Download the K most-frequent glosses (e.g. 100 for WLASL-100).",
    )
    group.add_argument(
        "--words-file",
        type=Path,
        help="Path to a newline-delimited file of gloss labels to download.",
    )
    parser.add_argument(
        "--limit-per-class",
        type=int,
        help="Cap instances per gloss (useful for smoke tests).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the selected glosses and instance counts, then exit.",
    )
    args = parser.parse_args()

    if args.top_k is None and args.words_file is None:
        args.top_k = 100  # sensible default for capstone scope

    meta = _require_metadata()
    selected = _select_glosses(meta, args.top_k, args.words_file)

    if not selected:
        sys.exit("No glosses matched the selection criteria.")

    print(f"Selected {len(selected)} gloss(es). Instance counts:")
    for entry in selected:
        print(f"  {entry['gloss']:20s}  {len(entry['instances']):4d}")

    if args.dry_run:
        print("\n--dry-run: no downloads performed.")
        return

    WLASL_ROOT.mkdir(parents=True, exist_ok=True)
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

    yt_dlp_cmd = _require_yt_dlp()
    counts = download_all(selected, yt_dlp_cmd, args.limit_per_class)

    print("\nDownload summary:")
    for status, n in counts.most_common():
        print(f"  {status:20s}  {n}")
    print(f"\nPer-instance log: {LOG_PATH}")
    print("Next step: python backend/scripts/audit_wlasl.py")


if __name__ == "__main__":
    main()
