"""Soft-vote ensemble evaluator for word-model runs sharing a vocabulary.

Loads each run's best Keras checkpoint, runs the full test split, averages
the softmax outputs, and reports test top-1 / top-5 of the ensemble plus
each member. Members MUST share the same `word_label_map.json` (i.e.
trained on the same `--words-file`).

Usage:
  python backend/scripts/ensemble_eval.py \
      --runs word-curated-v3-64cls word-curated-v3-64cls-seed1 \
             word-curated-v3-64cls-seed2 word-curated-v3-64cls-seed3 \
             word-curated-v3-64cls-seed4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from backend.data.constants import FEATURE_DIM


def _resample(arr: np.ndarray, target_T: int) -> np.ndarray:
    if arr.shape[0] == target_T:
        return arr.astype(np.float32)
    nz = np.any(arr != 0, axis=1)
    if not nz.any():
        return np.zeros((target_T, arr.shape[1]), dtype=np.float32)
    last = int(np.where(nz)[0].max()) + 1
    core = arr[:last]
    src_t = np.linspace(0.0, 1.0, last)
    dst_t = np.linspace(0.0, 1.0, target_T)
    out = np.empty((target_T, arr.shape[1]), dtype=np.float32)
    for d in range(arr.shape[1]):
        out[:, d] = np.interp(dst_t, src_t, core[:, d])
    return out


def _gloss_from_filename(path: Path) -> str:
    stem = path.stem.lower()
    parts = stem.split("_")
    if (len(parts) >= 3 and parts[-1].isdigit()
            and parts[-2].startswith("s") and len(parts[-2]) > 1):
        return "_".join(parts[:-2])
    return "_".join(parts[:-2]) if len(parts) >= 3 else stem


def _load_test_set(label_map: dict[str, int]) -> list[tuple[Path, int]]:
    DATA = REPO_ROOT / "data" / "processed" / "words" / "test"
    items = []
    for npy in DATA.glob("*.npy"):
        g = _gloss_from_filename(npy).lower()
        if g in label_map:
            items.append((npy, label_map[g]))
    return items


def _eval_one(run: str, items: list[tuple[Path, int]]) -> tuple[np.ndarray, dict]:
    import tensorflow as tf
    cfg = json.loads((REPO_ROOT / "artifacts/runs" / run / "reports/config.json").read_text())
    arch = cfg["arch"]
    seq_len = cfg["seq_len"]
    ckpt = REPO_ROOT / "artifacts/runs" / run / "checkpoints" / f"{arch}_best.keras"
    model = tf.keras.models.load_model(str(ckpt), compile=False)
    probs = np.zeros((len(items), cfg["n_classes"]), dtype=np.float32)
    BATCH = 32
    for s in range(0, len(items), BATCH):
        batch = items[s:s + BATCH]
        x = np.stack([_resample(np.load(p).astype(np.float32), seq_len) for p, _ in batch])
        probs[s:s + len(batch)] = model.predict(x, verbose=0)
    return probs, cfg


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out", default="artifacts/reports/word_ensemble.md")
    args = p.parse_args()

    # All runs must share the label map.
    label_maps = [json.loads((REPO_ROOT / "artifacts/runs" / r / "word_label_map.json").read_text())
                  for r in args.runs]
    assert all(m == label_maps[0] for m in label_maps), "runs do not share vocab"
    label_map = label_maps[0]
    inv = {v: k for k, v in label_map.items()}

    items = _load_test_set(label_map)
    y_true = np.array([y for _, y in items])
    print(f"Loaded {len(items)} test samples over {len(label_map)} classes")

    per_run = []
    sum_probs = None
    for r in args.runs:
        print(f"  evaluating {r}...")
        probs, cfg = _eval_one(r, items)
        pred = probs.argmax(axis=1)
        top5 = np.argsort(probs, axis=1)[:, -5:]
        top1 = (pred == y_true).mean()
        top5_acc = np.mean([y in row for y, row in zip(y_true, top5)])
        per_run.append({"run": r, "top1": float(top1), "top5": float(top5_acc)})
        print(f"    top1={top1:.4f}  top5={top5_acc:.4f}")
        sum_probs = probs if sum_probs is None else sum_probs + probs

    # Ensemble = mean softmax
    ens = sum_probs / len(args.runs)
    pred = ens.argmax(axis=1)
    top5 = np.argsort(ens, axis=1)[:, -5:]
    ens_top1 = (pred == y_true).mean()
    ens_top5 = np.mean([y in row for y, row in zip(y_true, top5)])

    print(f"\nENSEMBLE (n={len(args.runs)}):  top1={ens_top1:.4f}  top5={ens_top5:.4f}")

    md = [f"# Soft-vote ensemble — {len(args.runs)} members over {len(label_map)} classes",
          "",
          "| run | test top-1 | test top-5 |", "|---|---:|---:|"]
    for r in per_run:
        md.append(f"| `{r['run']}` | {r['top1']:.4f} | {r['top5']:.4f} |")
    md.append(f"| **ensemble (mean softmax)** | **{ens_top1:.4f}** | **{ens_top5:.4f}** |")
    md.append("")
    md.append(f"Gate (test top-1 ≥ 0.85 AND test top-5 ≥ 0.97): "
              f"**{'PASS' if (ens_top1 >= 0.85 and ens_top5 >= 0.97) else 'FAIL'}**")

    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md) + "\n")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
