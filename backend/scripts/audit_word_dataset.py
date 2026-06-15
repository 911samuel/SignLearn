"""Lightweight audit for the word landmark corpus (data/processed/words/).

Counts samples and unique signers per class per split, and clip-length
stats (sequences are pre-padded to seq_len so we report nonzero-frame
counts as a proxy for the "real" clip duration).

Output:
  artifacts/reports/word_dataset_audit.json
  artifacts/reports/word_dataset_audit.md
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

DATA_ROOT = REPO_ROOT / "data" / "processed" / "words"
REPORTS_DIR = REPO_ROOT / "artifacts" / "reports"

# Tolerant signer-token regex — WLASL uses s05, YouTube uses syt*, ASL
# Citizen uses sac*. We treat the whole token (minus leading 's') as the
# signer identity.
_SIGNER_TOK = re.compile(r"_(s[A-Za-z0-9]+)_\d+$")


def _parse(path: Path) -> tuple[str | None, str | None]:
    stem = path.stem.lower()
    m = _SIGNER_TOK.search(stem)
    if not m:
        return None, None
    signer = m.group(1)
    # gloss is everything before _<signer>_<idx>
    gloss = stem[: m.start()]
    return gloss, signer


def _nonzero_frames(arr: np.ndarray) -> int:
    # A frame is "active" if any landmark coord is nonzero.
    return int(np.any(arr != 0, axis=1).sum())


def audit() -> dict:
    out: dict = {"splits": {}}
    for split in ("train", "val", "test"):
        sd = DATA_ROOT / split
        if not sd.exists():
            continue
        per_class_counts: dict[str, int] = defaultdict(int)
        per_class_signers: dict[str, set[str]] = defaultdict(set)
        per_class_nonzero: dict[str, list[int]] = defaultdict(list)
        for npy in sd.glob("*.npy"):
            gloss, signer = _parse(npy)
            if gloss is None:
                continue
            per_class_counts[gloss] += 1
            if signer:
                per_class_signers[gloss].add(signer)
            try:
                arr = np.load(npy, mmap_mode="r")
                per_class_nonzero[gloss].append(_nonzero_frames(np.asarray(arr)))
            except Exception:
                pass
        out["splits"][split] = {
            cls: {
                "count": per_class_counts[cls],
                "signer_count": len(per_class_signers[cls]),
                "signers": sorted(per_class_signers[cls]),
                "nonzero_frames_mean": (
                    float(np.mean(per_class_nonzero[cls])) if per_class_nonzero[cls] else None
                ),
                "nonzero_frames_min": (
                    int(np.min(per_class_nonzero[cls])) if per_class_nonzero[cls] else None
                ),
                "nonzero_frames_max": (
                    int(np.max(per_class_nonzero[cls])) if per_class_nonzero[cls] else None
                ),
            }
            for cls in sorted(per_class_counts)
        }
    # Flat per-class view (keyed by class) using train stats — this is what
    # analyze_confusion.py consumes.
    flat: dict = {}
    train = out["splits"].get("train", {})
    val = out["splits"].get("val", {})
    test = out["splits"].get("test", {})
    classes = set(train) | set(val) | set(test)
    for c in sorted(classes):
        flat[c] = {
            "n_train": train.get(c, {}).get("count", 0),
            "n_val": val.get(c, {}).get("count", 0),
            "n_test": test.get(c, {}).get("count", 0),
            "signer_count": train.get(c, {}).get("signer_count", 0),
            "nonzero_frames_mean": train.get(c, {}).get("nonzero_frames_mean"),
        }
    out["per_class"] = flat
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(REPORTS_DIR / "word_dataset_audit.json"))
    args = p.parse_args()

    data = audit()
    out_json = Path(args.out)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(data, indent=2))
    log.info("Wrote %s", out_json)

    # Markdown summary.
    md = ["# Word dataset audit", "",
          "| class | n_train | n_val | n_test | signers | nz-frames (mean) |",
          "|---|---:|---:|---:|---:|---:|"]
    for c, row in data["per_class"].items():
        nf = row["nonzero_frames_mean"]
        md.append(
            f"| {c} | {row['n_train']} | {row['n_val']} | {row['n_test']} | "
            f"{row['signer_count']} | {nf:.1f} |" if nf is not None
            else f"| {c} | {row['n_train']} | {row['n_val']} | {row['n_test']} | "
                 f"{row['signer_count']} | — |"
        )
    out_md = REPORTS_DIR / "word_dataset_audit.md"
    out_md.write_text("\n".join(md) + "\n")
    log.info("Wrote %s", out_md)

    # Quick stats summary.
    under = [(c, r["n_train"], r["signer_count"])
             for c, r in data["per_class"].items()
             if r["n_train"] < 15 or r["signer_count"] < 3]
    log.info("Classes with n_train<15 OR signers<3: %d / %d",
             len(under), len(data["per_class"]))


if __name__ == "__main__":
    main()
