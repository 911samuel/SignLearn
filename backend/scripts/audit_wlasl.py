"""
Audit the downloaded WLASL corpus and recommend a final class list.

Reads ``data/raw/wlasl/`` (videos + download_log.jsonl + metadata) and writes:

  artifacts/reports/wlasl100_inventory.md    — human-readable report
  artifacts/reports/wlasl100_inventory.json  — machine-readable summary
  configs/wlasl_words_curated.txt            — recommended final gloss list

Checks
------
- Per-gloss download yield (ok / failed / no_url / skipped).
- Per-gloss usable clip count after dead-link filtering.
- Clip-length distribution (uses ffprobe; falls back to "unknown" if absent).
- Signer-disjoint split feasibility (does each gloss have ≥2 signers?).
- Recommended class list filtered by MIN_USABLE_CLIPS (default 15) and
  MIN_DISTINCT_SIGNERS (default 2).

Usage
-----
  python backend/scripts/audit_wlasl.py
  python backend/scripts/audit_wlasl.py --min-clips 20 --min-signers 3
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
WLASL_ROOT = REPO_ROOT / "data" / "raw" / "wlasl"
METADATA_PATH = WLASL_ROOT / "WLASL_v0.3.json"
LOG_PATH = WLASL_ROOT / "download_log.jsonl"
VIDEOS_DIR = WLASL_ROOT / "videos"

REPORTS_DIR = REPO_ROOT / "artifacts" / "reports"
CONFIGS_DIR = REPO_ROOT / "configs"
MD_OUT = REPORTS_DIR / "wlasl100_inventory.md"
JSON_OUT = REPORTS_DIR / "wlasl100_inventory.json"
CURATED_OUT = CONFIGS_DIR / "wlasl_words_curated.txt"


def _load_log() -> list[dict]:
    if not LOG_PATH.exists():
        return []
    return [json.loads(line) for line in LOG_PATH.read_text().splitlines() if line.strip()]


def _load_metadata() -> dict[str, dict]:
    if not METADATA_PATH.exists():
        return {}
    with METADATA_PATH.open() as f:
        meta = json.load(f)
    return {entry["gloss"]: entry for entry in meta}


def _probe_duration_frames(path: Path) -> tuple[float | None, int | None]:
    """Return (duration_seconds, nb_frames) for a video, or (None, None) on failure."""
    if not shutil.which("ffprobe"):
        return None, None
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=nb_frames,duration,r_frame_rate",
        "-of", "json",
        str(path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    except subprocess.TimeoutExpired:
        return None, None
    if result.returncode != 0:
        return None, None
    try:
        data = json.loads(result.stdout)
        stream = data["streams"][0]
    except (KeyError, IndexError, json.JSONDecodeError):
        return None, None
    duration = float(stream["duration"]) if stream.get("duration") else None
    nb_frames = int(stream["nb_frames"]) if stream.get("nb_frames", "0").isdigit() else None
    if nb_frames is None and duration is not None and stream.get("r_frame_rate"):
        num, _, den = stream["r_frame_rate"].partition("/")
        try:
            fps = float(num) / float(den) if den else float(num)
            nb_frames = int(round(duration * fps))
        except (ValueError, ZeroDivisionError):
            nb_frames = None
    return duration, nb_frames


def _summarize(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    s = sorted(values)
    return {
        "n": len(values),
        "min": s[0],
        "p10": s[max(0, len(s) // 10)],
        "median": statistics.median(s),
        "p90": s[min(len(s) - 1, (len(s) * 9) // 10)],
        "max": s[-1],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit downloaded WLASL corpus.")
    parser.add_argument("--min-clips", type=int, default=15,
                        help="Minimum usable clips required to keep a gloss (default 15).")
    parser.add_argument("--min-signers", type=int, default=2,
                        help="Minimum distinct signers required to keep a gloss (default 2).")
    parser.add_argument("--no-ffprobe", action="store_true",
                        help="Skip clip-length probing (faster, less info).")
    args = parser.parse_args()

    if not VIDEOS_DIR.exists():
        print(f"No videos directory at {VIDEOS_DIR}. Run download_wlasl.py first.")
        return

    log = _load_log()
    metadata = _load_metadata()

    # Per-gloss aggregates from the download log.
    status_by_gloss: dict[str, Counter] = defaultdict(Counter)
    signers_by_gloss: dict[str, set] = defaultdict(set)
    for rec in log:
        gloss = rec["gloss"]
        status_by_gloss[gloss][rec["status"]] += 1
        if rec["status"] in ("ok", "skipped_existing") and rec.get("signer_id") is not None:
            signers_by_gloss[gloss].add(rec["signer_id"])

    # Cross-check actual files on disk (truth for "usable").
    on_disk: dict[str, list[Path]] = {}
    for gloss_dir in sorted(VIDEOS_DIR.iterdir()):
        if gloss_dir.is_dir():
            files = sorted(p for p in gloss_dir.glob("*.mp4") if p.stat().st_size > 0)
            on_disk[gloss_dir.name] = files

    glosses = sorted(set(on_disk) | set(status_by_gloss))

    # Probe clip lengths once per file.
    all_frame_counts: list[int] = []
    per_gloss_frames: dict[str, list[int]] = defaultdict(list)
    if not args.no_ffprobe:
        for gloss, files in on_disk.items():
            for f in files:
                _, frames = _probe_duration_frames(f)
                if frames:
                    per_gloss_frames[gloss].append(frames)
                    all_frame_counts.append(frames)

    # Build per-gloss report rows.
    rows = []
    for gloss in glosses:
        files = on_disk.get(gloss, [])
        meta_entry = metadata.get(gloss, {})
        total_listed = len(meta_entry.get("instances", []))
        statuses = status_by_gloss.get(gloss, Counter())
        usable = len(files)
        signers = len(signers_by_gloss.get(gloss, set()))
        rows.append({
            "gloss": gloss,
            "listed": total_listed,
            "ok": statuses.get("ok", 0) + statuses.get("skipped_existing", 0),
            "failed": statuses.get("download_failed", 0) + statuses.get("timeout", 0),
            "no_url": statuses.get("no_url", 0),
            "usable_on_disk": usable,
            "distinct_signers": signers,
            "frame_stats": _summarize([float(x) for x in per_gloss_frames.get(gloss, [])]),
        })

    # Filter to recommended class list.
    kept = [
        r for r in rows
        if r["usable_on_disk"] >= args.min_clips
        and r["distinct_signers"] >= args.min_signers
    ]
    kept.sort(key=lambda r: r["usable_on_disk"], reverse=True)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)

    # JSON summary.
    summary = {
        "thresholds": {"min_clips": args.min_clips, "min_signers": args.min_signers},
        "totals": {
            "glosses_attempted": len(rows),
            "glosses_kept": len(kept),
            "clips_on_disk": sum(r["usable_on_disk"] for r in rows),
            "clips_kept": sum(r["usable_on_disk"] for r in kept),
        },
        "overall_frame_stats": _summarize([float(x) for x in all_frame_counts]),
        "rows": rows,
        "kept_glosses": [r["gloss"] for r in kept],
    }
    JSON_OUT.write_text(json.dumps(summary, indent=2))

    # Markdown report.
    lines: list[str] = []
    lines.append("# WLASL Inventory")
    lines.append("")
    lines.append(f"- Glosses attempted: **{summary['totals']['glosses_attempted']}**")
    lines.append(f"- Glosses kept (≥{args.min_clips} clips, ≥{args.min_signers} signers): "
                 f"**{summary['totals']['glosses_kept']}**")
    lines.append(f"- Total usable clips on disk: **{summary['totals']['clips_on_disk']}**")
    lines.append(f"- Clips in kept glosses: **{summary['totals']['clips_kept']}**")
    fs = summary["overall_frame_stats"]
    if fs.get("n"):
        lines.append(
            f"- Clip-length frames — median **{fs['median']:.0f}**, "
            f"p10 {fs['p10']:.0f}, p90 {fs['p90']:.0f}, "
            f"min {fs['min']:.0f}, max {fs['max']:.0f} (n={fs['n']})"
        )
        lines.append("")
        lines.append(f"**Suggested sequence length:** ~{int(fs['p90'])} frames "
                     f"(covers 90% of clips without excessive padding).")
    lines.append("")
    lines.append("## Per-gloss table")
    lines.append("")
    lines.append("| Gloss | Listed | OK | Failed | No URL | On disk | Signers | Median frames |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in sorted(rows, key=lambda x: x["usable_on_disk"], reverse=True):
        median_frames = (f"{r['frame_stats']['median']:.0f}"
                         if r['frame_stats'].get('n') else "—")
        lines.append(
            f"| {r['gloss']} | {r['listed']} | {r['ok']} | {r['failed']} | "
            f"{r['no_url']} | {r['usable_on_disk']} | {r['distinct_signers']} | "
            f"{median_frames} |"
        )
    lines.append("")
    lines.append("## Recommended class list")
    lines.append("")
    lines.append(f"Written to `{CURATED_OUT.relative_to(REPO_ROOT)}` — "
                 f"{len(kept)} gloss(es). Review and prune to match your "
                 f"proposal's everyday-phrase focus before training.")
    MD_OUT.write_text("\n".join(lines) + "\n")

    # Curated word list.
    CURATED_OUT.write_text("\n".join(r["gloss"] for r in kept) + "\n")

    print(f"Wrote {MD_OUT}")
    print(f"Wrote {JSON_OUT}")
    print(f"Wrote {CURATED_OUT}  ({len(kept)} gloss(es))")


if __name__ == "__main__":
    main()
