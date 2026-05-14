"""Phase 2 — Dataset audit for the SignLearn landmark corpus.

Audits ``data/processed/{train,val,test}`` and produces two artifacts:

  artifacts/reports/dataset_audit.json   — machine-readable summary
  artifacts/reports/dataset_audit.md     — human-readable report

Checks performed
----------------
- Per-split, per-class sample counts (flags classes with < MIN_SAMPLES)
- Sequence shape + dtype validation (must be (30, 126) float32-castable)
- Zero-frame ratio per sample (proxy for "no hand detected" frames)
- Post-normalization landmark range sanity (bounded roughly to [-1, 1])
- Duplicate detection via SHA-256 over the flattened raw arrays
- Cross-split label-leakage check (same content hash in two splits)
- Class-balance Gini coefficient per split

Usage
-----
python backend/scripts/audit_dataset.py
python backend/scripts/audit_dataset.py --data-dir data/processed --out artifacts/reports
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# Filename convention: <label>_s<signer>_<idx>.npy (e.g. hello_s02_0042.npy).
# Allow the label itself to contain underscores (e.g. thank_you).
_SIGNER_RE = re.compile(r"_s(\d+)_\d+$")


def _extract_signer_id(stem: str) -> str | None:
    """Return the signer ID embedded in a filename stem, or None if absent.

    Conservative: matches only the strict `_s<digits>_<digits>` suffix so
    legacy filenames without explicit signer IDs (eg. MD5-pseudo-subject
    files from the Kaggle alphabet ingest) are reported as `None` rather
    than silently misattributed.
    """
    m = _SIGNER_RE.search(stem)
    return m.group(1) if m else None

_REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from backend.data.constants import FEATURE_DIM, SEQUENCE_LEN
from backend.data.dataset import list_split
from backend.data.label_map import inverse_label_map
from backend.data.normalize import normalize_sequence
from backend.model.config import PROCESSED_DIR, REPORTS_DIR

MIN_SAMPLES_PER_CLASS = 100
NORMALIZED_RANGE = 1.5  # post-normalize values should sit well within ±1.5


def _sha256(arr: np.ndarray) -> str:
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _gini(values: list[int]) -> float:
    """Gini coefficient over class-frequency counts. 0 = perfectly balanced."""
    if not values:
        return 0.0
    n = len(values)
    s = sum(values)
    if s == 0:
        return 0.0
    sorted_v = sorted(values)
    cum = sum((i + 1) * v for i, v in enumerate(sorted_v))
    return (2 * cum) / (n * s) - (n + 1) / n


def _zero_frame_ratio(seq: np.ndarray) -> float:
    return float(np.mean(np.all(seq == 0, axis=1)))


def _audit_split(split: str, data_dir: Path) -> dict:
    items = list_split(split, processed_dir=data_dir)
    inv = inverse_label_map()

    per_class_counts: dict[str, int] = {}
    sample_records: list[dict] = []
    seq_lens: list[int] = []
    zero_ratios: list[float] = []
    bad_shape: list[str] = []
    out_of_range: list[str] = []

    for path, label_idx in items:
        name = inv.get(label_idx, f"unk_{label_idx}")
        per_class_counts[name] = per_class_counts.get(name, 0) + 1

        try:
            arr = np.load(path).astype(np.float32)
        except Exception as exc:  # noqa: BLE001
            bad_shape.append(f"{path.name}: load error {exc}")
            continue

        if arr.ndim != 2 or arr.shape[1] != FEATURE_DIM:
            bad_shape.append(f"{path.name}: shape {arr.shape}")
            continue
        seq_lens.append(arr.shape[0])

        zero_ratios.append(_zero_frame_ratio(arr))

        # Post-normalize range check.
        try:
            norm = normalize_sequence(arr) if arr.shape[0] == SEQUENCE_LEN else None
        except Exception as exc:  # noqa: BLE001
            bad_shape.append(f"{path.name}: normalize error {exc}")
            norm = None
        if norm is not None and np.max(np.abs(norm)) > NORMALIZED_RANGE:
            out_of_range.append(f"{path.name}: |max|={float(np.max(np.abs(norm))):.3f}")

        sample_records.append({
            "path": str(path.relative_to(data_dir)),
            "label": name,
            "hash": _sha256(arr),
            "zero_frame_ratio": _zero_frame_ratio(arr),
            "signer_id": _extract_signer_id(path.stem),
        })

    counts = list(per_class_counts.values())
    under = sorted(
        [(c, n) for c, n in per_class_counts.items() if n < MIN_SAMPLES_PER_CLASS],
        key=lambda kv: kv[1],
    )

    # Duplicate detection (same content hash within split)
    hash_groups: dict[str, list[str]] = {}
    for rec in sample_records:
        hash_groups.setdefault(rec["hash"], []).append(rec["path"])
    dupes_within = [paths for paths in hash_groups.values() if len(paths) > 1]

    # Signer-level statistics: how many distinct signers, per-signer sample
    # counts, per-signer zero-frame mean (useful for spotting bad recordings).
    per_signer_counts: dict[str, int] = defaultdict(int)
    per_signer_zero_mean: dict[str, list[float]] = defaultdict(list)
    n_unknown_signer = 0
    for rec in sample_records:
        sid = rec["signer_id"]
        if sid is None:
            n_unknown_signer += 1
            continue
        per_signer_counts[sid] += 1
        per_signer_zero_mean[sid].append(rec["zero_frame_ratio"])
    per_signer_summary = {
        sid: {
            "count": per_signer_counts[sid],
            "zero_frame_mean": round(statistics.mean(per_signer_zero_mean[sid]), 4),
        }
        for sid in sorted(per_signer_counts)
    }

    return {
        "split": split,
        "n_samples": len(sample_records),
        "n_classes_present": len(per_class_counts),
        "per_class_counts": dict(sorted(per_class_counts.items())),
        "under_minimum": [{"class": c, "count": n} for c, n in under],
        "min_samples_threshold": MIN_SAMPLES_PER_CLASS,
        "gini_coefficient": round(_gini(counts), 4) if counts else 0.0,
        "seq_len_stats": {
            "min": min(seq_lens) if seq_lens else 0,
            "max": max(seq_lens) if seq_lens else 0,
            "mean": round(statistics.mean(seq_lens), 2) if seq_lens else 0.0,
            "expected": SEQUENCE_LEN,
            "non_conforming": sum(1 for L in seq_lens if L != SEQUENCE_LEN),
        },
        "zero_frame_ratio": {
            "mean": round(statistics.mean(zero_ratios), 4) if zero_ratios else 0.0,
            "max":  round(max(zero_ratios), 4) if zero_ratios else 0.0,
            "n_all_zero_samples": sum(1 for r in zero_ratios if r == 1.0),
        },
        "bad_shape": bad_shape,
        "out_of_range_after_normalize": out_of_range,
        "duplicates_within_split": dupes_within,
        "per_signer_summary": per_signer_summary,
        "n_unknown_signer": n_unknown_signer,
        "_samples": sample_records,  # private, used for cross-split leakage check
    }


def _cross_split_leakage(splits: list[dict]) -> list[dict]:
    """Find samples whose content hash appears in more than one split."""
    by_hash: dict[str, list[tuple[str, str]]] = {}
    for s in splits:
        for rec in s["_samples"]:
            by_hash.setdefault(rec["hash"], []).append((s["split"], rec["path"]))
    leaks = []
    for h, locs in by_hash.items():
        unique_splits = {sp for sp, _ in locs}
        if len(unique_splits) > 1:
            leaks.append({"hash": h, "occurrences": locs})
    return leaks


def per_signer_leakage(splits: list[dict]) -> dict:
    """Detect signer IDs that appear in more than one split.

    Pure signer-leakage check: a model that learns signer-specific quirks will
    score higher on val/test than it would on unseen people if any signer ID
    appears in multiple splits. Returns the offenders + a clean per-split
    signer set summary suitable for an `audit_signers.md` report.

    Returns
    -------
    {
        "split_signers":  {split_name: sorted([signer_ids])},
        "overlaps":       [{"signer": sid, "splits": ["train", "val"]}, ...],
        "n_unknown":      {split_name: int}  # samples without parseable signer ID
    }
    """
    split_signers: dict[str, set[str]] = {}
    n_unknown: dict[str, int] = {}
    for s in splits:
        signers = {rec["signer_id"] for rec in s["_samples"] if rec["signer_id"]}
        split_signers[s["split"]] = signers
        n_unknown[s["split"]] = sum(1 for rec in s["_samples"] if rec["signer_id"] is None)

    overlaps: list[dict] = []
    all_ids: set[str] = set().union(*split_signers.values())
    for sid in sorted(all_ids):
        occ = [name for name, ids in split_signers.items() if sid in ids]
        if len(occ) > 1:
            overlaps.append({"signer": sid, "splits": occ})

    return {
        "split_signers": {k: sorted(v) for k, v in split_signers.items()},
        "overlaps": overlaps,
        "n_unknown": n_unknown,
    }


def render_signer_markdown(signer_report: dict) -> str:
    """Render the per-signer leakage report as standalone markdown."""
    lines = ["# SignLearn — Per-Signer Leakage Audit", ""]
    overlaps = signer_report["overlaps"]
    if overlaps:
        lines.append(f"## ⚠️ {len(overlaps)} signer ID(s) leak across splits")
        lines.append("")
        lines.append("| Signer | Splits |")
        lines.append("|---|---|")
        for o in overlaps:
            lines.append(f"| `{o['signer']}` | {', '.join(o['splits'])} |")
        lines.append("")
        lines.append(
            "**Fix:** remove the offending files from val/test, or re-assign "
            "by editing `_SIGNER_SPLIT` in `extract.py`. A signer who appears "
            "in train *and* val will inflate val accuracy by the model "
            "memorizing their idiosyncratic landmark patterns."
        )
        lines.append("")
    else:
        lines.append("✅ No signer leakage detected across train/val/test.")
        lines.append("")

    lines.append("## Per-split signer inventory")
    lines.append("")
    for split, ids in signer_report["split_signers"].items():
        unknown = signer_report["n_unknown"].get(split, 0)
        lines.append(f"### `{split}` — {len(ids)} signer(s), {unknown} sample(s) with unknown signer")
        lines.append("")
        if ids:
            lines.append(f"Signer IDs: {', '.join(f'`{s}`' for s in ids)}")
        else:
            lines.append("(no signer IDs parsed)")
        lines.append("")
    return "\n".join(lines)


def _recommendation_block(splits: list[dict]) -> dict:
    """Translate findings into actionable next steps."""
    issues = []
    advice = []
    train = next((s for s in splits if s["split"] == "train"), None)
    if train is None or train["n_samples"] == 0:
        issues.append("No training samples found.")
    else:
        if train["under_minimum"]:
            issues.append(
                f"{len(train['under_minimum'])} train class(es) below "
                f"{MIN_SAMPLES_PER_CLASS} samples"
            )
            advice.append(
                "Top up underrepresented classes via `extract_landmarks.py` "
                "before training to reduce class-imbalance bias."
            )
        if train["gini_coefficient"] > 0.2:
            issues.append(f"Train Gini {train['gini_coefficient']} > 0.2 (imbalanced)")
            advice.append(
                "Consider class-weighted loss or oversampling minority classes."
            )
        if train["zero_frame_ratio"]["mean"] > 0.2:
            issues.append(
                f"Mean zero-frame ratio is {train['zero_frame_ratio']['mean']:.2%} — "
                "many frames have no detected hand."
            )
            advice.append(
                "Investigate MediaPipe extraction quality; consider Savitzky-Golay "
                "smoothing or stricter detection thresholds during capture."
            )

    return {"issues": issues, "advice": advice}


def render_markdown(report: dict) -> str:
    lines = ["# SignLearn — Dataset Audit", ""]
    lines.append(f"- Total splits scanned: **{len(report['splits'])}**")
    lines.append(f"- Cross-split label leaks: **{len(report['cross_split_leakage'])}**")
    lines.append("")

    for s in report["splits"]:
        lines.append(f"## Split: `{s['split']}`")
        lines.append("")
        lines.append(f"- Samples: **{s['n_samples']}**")
        lines.append(f"- Classes present: **{s['n_classes_present']}**")
        lines.append(f"- Gini coefficient: **{s['gini_coefficient']}** (0 = balanced)")
        lines.append(
            f"- Sequence length: min={s['seq_len_stats']['min']}, "
            f"max={s['seq_len_stats']['max']}, "
            f"expected={s['seq_len_stats']['expected']}, "
            f"non-conforming={s['seq_len_stats']['non_conforming']}"
        )
        lines.append(
            f"- Zero-frame ratio: mean={s['zero_frame_ratio']['mean']:.4f}, "
            f"max={s['zero_frame_ratio']['max']:.4f}, "
            f"all-zero samples={s['zero_frame_ratio']['n_all_zero_samples']}"
        )
        lines.append(f"- Within-split duplicates: **{len(s['duplicates_within_split'])}**")
        lines.append(
            f"- Post-normalize out-of-range samples: **"
            f"{len(s['out_of_range_after_normalize'])}**"
        )
        lines.append(f"- Bad-shape / load errors: **{len(s['bad_shape'])}**")
        lines.append("")
        lines.append("| Class | Count |")
        lines.append("|---|---|")
        for cls, n in s["per_class_counts"].items():
            flag = " ⚠️" if n < MIN_SAMPLES_PER_CLASS else ""
            lines.append(f"| {cls} | {n}{flag} |")
        lines.append("")

    leaks = report["cross_split_leakage"]
    if leaks:
        lines.append("## ⚠️ Cross-split leakage")
        lines.append("")
        for leak in leaks[:20]:
            occ = ", ".join(f"{sp}:{p}" for sp, p in leak["occurrences"])
            lines.append(f"- {occ}")
        if len(leaks) > 20:
            lines.append(f"- … and {len(leaks) - 20} more")
        lines.append("")

    rec = report["recommendation"]
    lines.append("## Recommendations")
    lines.append("")
    if rec["issues"]:
        lines.append("**Issues detected:**")
        for i in rec["issues"]:
            lines.append(f"- {i}")
        lines.append("")
    else:
        lines.append("No blocking issues detected.")
        lines.append("")
    if rec["advice"]:
        lines.append("**Suggested actions:**")
        for a in rec["advice"]:
            lines.append(f"- {a}")
        lines.append("")
    return "\n".join(lines)


def audit(data_dir: Path, out_dir: Path) -> dict:
    splits = [_audit_split(s, data_dir) for s in ("train", "val", "test")]
    leakage = _cross_split_leakage(splits)
    signer_leakage = per_signer_leakage(splits)
    # Strip the private _samples blob from the JSON output to keep it small.
    for s in splits:
        s.pop("_samples", None)
    report = {
        "data_dir": str(data_dir),
        "splits": splits,
        "cross_split_leakage": leakage,
        "signer_leakage": signer_leakage,
        "recommendation": _recommendation_block(splits),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "dataset_audit.json"
    md_path   = out_dir / "dataset_audit.md"
    signer_md = out_dir / "audit_signers.md"
    json_path.write_text(json.dumps(report, indent=2))
    md_path.write_text(render_markdown(report))
    signer_md.write_text(render_signer_markdown(signer_leakage))
    print(f"JSON       → {json_path}")
    print(f"MD         → {md_path}")
    print(f"Signer MD  → {signer_md}")
    if signer_leakage["overlaps"]:
        print(
            f"⚠️  {len(signer_leakage['overlaps'])} signer ID(s) leak across splits — "
            f"see {signer_md}"
        )
    return report


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Audit the SignLearn processed dataset")
    p.add_argument("--data-dir", type=Path, default=PROCESSED_DIR)
    p.add_argument("--out",      type=Path, default=REPORTS_DIR)
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    audit(args.data_dir, args.out)
