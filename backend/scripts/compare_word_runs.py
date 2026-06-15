"""Compare a set of word-model runs.

Reads each run's reports/{config.json, test_metrics.json, per_class_accuracy.json}
and writes a single comparison markdown.

Usage:
  python backend/scripts/compare_word_runs.py \
      --runs word78-aug__... word78-aug__... \
      --out artifacts/reports/sweeps/word78-aug_compare.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
RUNS = REPO_ROOT / "artifacts" / "runs"


def _row(run: str) -> dict:
    d = RUNS / run
    try:
        cfg = json.loads((d / "reports" / "config.json").read_text())
    except Exception:
        cfg = {}
    try:
        tm = json.loads((d / "reports" / "test_metrics.json").read_text())
    except Exception:
        tm = {}
    try:
        pc = json.loads((d / "reports" / "per_class_accuracy.json").read_text())
        accs = [v["acc"] for v in pc.values() if v.get("acc") is not None]
        min_acc = min(accs) if accs else None
        below50 = sum(1 for a in accs if a < 0.5)
    except Exception:
        min_acc, below50 = None, None
    return {
        "run": run,
        "arch": cfg.get("arch"),
        "aug": cfg.get("aug_profile"),
        "fm": cfg.get("feature_mode", "raw"),
        "T": cfg.get("seq_len"),
        "epochs": cfg.get("epochs_run"),
        "top1": tm.get("accuracy"),
        "top5": tm.get("top5_acc"),
        "loss": tm.get("loss"),
        "min_class_acc": min_acc,
        "n_below_50pct": below50,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    rows = [_row(r) for r in args.runs]
    rows.sort(key=lambda r: (r["top1"] or 0), reverse=True)

    md = ["# Word run comparison", "",
          "| run | arch | aug | fm | T | epochs | top-1 | top-5 | min cls acc | <50% classes |",
          "|---|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        md.append(
            f"| `{r['run']}` | {r['arch']} | {r['aug']} | {r['fm']} | "
            f"{r['T']} | {r['epochs']} | "
            f"{r['top1']:.4f} | {r['top5']:.4f} | "
            .replace("None", "—")
            + (f"{r['min_class_acc']:.2f} | " if r['min_class_acc'] is not None else "— | ")
            + (f"{r['n_below_50pct']} |" if r['n_below_50pct'] is not None else "— |")
        )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md) + "\n")
    print(f"Wrote {out}")
    for r in rows:
        print(f"  {r['run']:60s}  top1={r['top1']:.4f}  top5={r['top5']:.4f}")


if __name__ == "__main__":
    main()
