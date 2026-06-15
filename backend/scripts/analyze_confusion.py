"""Confusion analysis for word-level runs.

Loads a trained Keras checkpoint, runs it over data/processed/words/test,
computes the test confusion matrix, and writes:

  artifacts/runs/<run>/reports/confusion_matrix.json
  artifacts/runs/<run>/reports/confusion_matrix.png
  artifacts/reports/word<N>_confusion_pairs.md

The markdown groups off-diagonal pairs (rate > --threshold, default 0.15)
into "linguistic twins" (manually curated pairs) and "data twins" (a class
in the pair has n_train < 15 OR signer_count < 3, per dataset_audit.json).

Usage:
  python backend/scripts/analyze_confusion.py \
      --run word-aslc-tcn-78cls-v1 \
      --threshold 0.15
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from backend.data.constants import FEATURE_DIM

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

DATA_ROOT = REPO_ROOT / "data" / "processed" / "words"
RUNS_DIR = REPO_ROOT / "artifacts" / "runs"
REPORTS_DIR = REPO_ROOT / "artifacts" / "reports"

# Pairs the user flagged as linguistically similar in the prompt + a few
# well-known ASL minimal-pair candidates. The annotation column in the
# output table is "linguistic" if the pair (or its reverse) appears here.
LINGUISTIC_PAIRS: set[frozenset[str]] = {
    frozenset({"help", "sorry"}),
    frozenset({"hear", "listen"}),
    frozenset({"hearing", "listen"}),
    frozenset({"talk", "say"}),
    frozenset({"nice", "clean"}),
    frozenset({"like", "love"}),
    # Common ASL minimal pairs the literature flags:
    frozenset({"mother", "father"}),
    frozenset({"good", "thank_you"}),
    frozenset({"yes", "knock"}),
    frozenset({"come", "go"}),
    frozenset({"deaf1", "deaf2"}),
    frozenset({"drink1", "drink2"}),
    frozenset({"eat1", "eat2"}),
    frozenset({"fine1", "fine2"}),
    frozenset({"hospital1", "hospital2"}),
    frozenset({"how1", "how2"}),
}


def _gloss_from_filename(path: Path) -> str:
    stem = path.stem.lower()
    parts = stem.split("_")
    if (len(parts) >= 3 and parts[-1].isdigit()
            and parts[-2].startswith("s") and len(parts[-2]) > 1):
        return "_".join(parts[:-2])
    return "_".join(parts[:-2]) if len(parts) >= 3 else stem


def load_test_set(label_map: dict[str, int], seq_len: int):
    items = []
    split_dir = DATA_ROOT / "test"
    for npy in split_dir.glob("*.npy"):
        gloss = _gloss_from_filename(npy).lower()
        if gloss not in label_map:
            continue
        arr = np.load(npy, mmap_mode="r")
        if arr.shape != (seq_len, FEATURE_DIM):
            continue
        items.append((npy, label_map[gloss]))
    return items


def load_audit(audit_path: Path) -> dict[str, dict]:
    """Return {class_name: {n_train, signer_count, ...}} from dataset_audit.json.

    Falls back to an empty dict if the audit isn't available — the data-twin
    tagging is then skipped (but the file is still produced).
    """
    if not audit_path.exists():
        log.warning("Audit JSON not found at %s — n_train/signers unknown", audit_path)
        return {}
    raw = json.loads(audit_path.read_text())
    # The audit shape varies; try a few common layouts defensively.
    if isinstance(raw, dict) and "classes" in raw:
        return {c["name"]: c for c in raw["classes"]}
    if isinstance(raw, dict) and "per_class" in raw:
        return raw["per_class"]
    if isinstance(raw, dict):
        return raw
    return {}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="Run name under artifacts/runs/")
    p.add_argument("--threshold", type=float, default=0.15,
                   help="Off-diagonal rate above which to flag a pair (default 0.15)")
    p.add_argument("--audit", default="artifacts/reports/dataset_audit.json",
                   help="Path to dataset audit JSON (optional)")
    args = p.parse_args()

    run_dir = RUNS_DIR / args.run
    cfg = json.loads((run_dir / "reports" / "config.json").read_text())
    label_map = json.loads((run_dir / "word_label_map.json").read_text())
    inv = {v: k for k, v in label_map.items()}
    n_classes = cfg["n_classes"]
    seq_len = cfg["seq_len"]
    arch = cfg["arch"]

    ckpt = run_dir / "checkpoints" / f"{arch}_best.keras"
    log.info("Loading checkpoint %s", ckpt)
    import tensorflow as tf
    model = tf.keras.models.load_model(str(ckpt), compile=False)

    items = load_test_set(label_map, seq_len)
    log.info("Test items: %d", len(items))

    # Batched predict.
    BATCH = 32
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    per_class_total = np.zeros(n_classes, dtype=np.int64)
    for start in range(0, len(items), BATCH):
        batch = items[start:start + BATCH]
        x = np.stack([np.load(p).astype(np.float32) for p, _ in batch])
        y = np.array([y for _, y in batch], dtype=np.int64)
        probs = model.predict(x, verbose=0)
        pred = np.argmax(probs, axis=1)
        for t, p_ in zip(y, pred):
            cm[t, p_] += 1
            per_class_total[t] += 1

    # Save raw matrix.
    out_json = run_dir / "reports" / "confusion_matrix.json"
    out_json.write_text(json.dumps({
        "labels": [inv[i] for i in range(n_classes)],
        "matrix": cm.tolist(),
    }, indent=2))
    log.info("Wrote %s", out_json)

    # Optional PNG.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(max(8, n_classes * 0.18), max(8, n_classes * 0.18)))
        with np.errstate(invalid="ignore", divide="ignore"):
            norm = cm.astype(float) / np.maximum(per_class_total[:, None], 1)
        ax.imshow(norm, vmin=0, vmax=1, cmap="magma")
        labels = [inv[i] for i in range(n_classes)]
        ax.set_xticks(range(n_classes)); ax.set_yticks(range(n_classes))
        ax.set_xticklabels(labels, rotation=90, fontsize=5)
        ax.set_yticklabels(labels, fontsize=5)
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        ax.set_title(f"{args.run} — normalized confusion (test)")
        fig.tight_layout()
        png = run_dir / "reports" / "confusion_matrix.png"
        fig.savefig(png, dpi=150)
        plt.close(fig)
        log.info("Wrote %s", png)
    except Exception as e:
        log.warning("Could not render PNG: %s", e)

    # Extract confused pairs.
    pairs = []
    for i in range(n_classes):
        total = per_class_total[i]
        if total == 0:
            continue
        for j in range(n_classes):
            if i == j:
                continue
            rate = cm[i, j] / total
            if rate >= args.threshold:
                pairs.append({
                    "true": inv[i], "pred": inv[j],
                    "rate": float(rate), "count": int(cm[i, j]),
                    "n_true": int(total),
                })
    pairs.sort(key=lambda p: p["rate"], reverse=True)

    # Audit join.
    audit = load_audit(REPO_ROOT / args.audit)

    def class_meta(name: str) -> tuple[int | None, int | None]:
        rec = audit.get(name) or audit.get(name.lower())
        if not rec:
            return None, None
        n_train = rec.get("n_train") or rec.get("train") or rec.get("counts", {}).get("train")
        signers = rec.get("signer_count") or rec.get("n_signers") or rec.get("signers")
        if isinstance(signers, (list, tuple, set)):
            signers = len(signers)
        return n_train, signers

    def tag_pair(a: str, b: str) -> str:
        is_ling = frozenset({a, b}) in LINGUISTIC_PAIRS
        na, sa = class_meta(a); nb, sb = class_meta(b)
        under = False
        for n, s in ((na, sa), (nb, sb)):
            if (n is not None and n < 15) or (s is not None and s < 3):
                under = True
                break
        if is_ling and under:
            return "mixed"
        if is_ling:
            return "linguistic"
        if under:
            return "data-twin"
        return "unknown"

    md_lines = [
        f"# Word78 Confusion Pairs (threshold = {args.threshold:.0%})",
        "",
        f"Run: `{args.run}`  ·  test top-1 from `reports/test_metrics.json`",
        f"Total flagged pairs: **{len(pairs)}**",
        "",
        "Tags:",
        "- **linguistic** — handshape/motion overlap (curated list in `analyze_confusion.py`)",
        "- **data-twin** — at least one class has n_train < 15 OR signer_count < 3",
        "- **mixed** — both",
        "- **unknown** — neither (worth manual inspection)",
        "",
        "| true | predicted as | rate | count / n_true | n_train(true) | signers(true) | n_train(pred) | signers(pred) | tag |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for pr in pairs:
        na, sa = class_meta(pr["true"]); nb, sb = class_meta(pr["pred"])
        tag = tag_pair(pr["true"], pr["pred"])
        md_lines.append(
            f"| {pr['true']} | {pr['pred']} | {pr['rate']:.0%} | "
            f"{pr['count']}/{pr['n_true']} | "
            f"{na if na is not None else '—'} | {sa if sa is not None else '—'} | "
            f"{nb if nb is not None else '—'} | {sb if sb is not None else '—'} | {tag} |"
        )

    # Per-class audit table (classes appearing in any flagged pair).
    classes_seen = sorted({pr["true"] for pr in pairs} | {pr["pred"] for pr in pairs})
    if classes_seen:
        md_lines += [
            "",
            "## Per-class audit (classes appearing in any flagged pair)",
            "",
            "| class | n_train | signer_count |",
            "|---|---:|---:|",
        ]
        for c in classes_seen:
            n, s = class_meta(c)
            md_lines.append(f"| {c} | {n if n is not None else '—'} | {s if s is not None else '—'} |")

    out_md = REPORTS_DIR / f"word{n_classes}_confusion_pairs.md"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md_lines) + "\n")
    log.info("Wrote %s", out_md)


if __name__ == "__main__":
    main()
