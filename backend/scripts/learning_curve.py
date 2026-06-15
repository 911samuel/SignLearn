"""Per-class learning curve + recording-plan generator (Step 6).

For each of the 15 worst classes from the current best run, train the best
config on {25, 50, 75, 100}% of that class's training samples (all other
classes at full strength), measure per-class test accuracy, fit a power
law per class, and predict the number of samples needed to reach a
target per-class accuracy. Output:

  artifacts/reports/data_collection_plan_v2.md

This is HEAVY: 15 × 4 = 60 short training runs. Set ``--epochs`` low
(default 50) and ``--worst-n`` to constrain.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).parent.parent.parent
RUNS = REPO_ROOT / "artifacts" / "runs"
REPORTS = REPO_ROOT / "artifacts" / "reports"

# Best-config defaults from steps 3–5.
BASE_CONFIG = dict(
    arch="tcn",
    seq_len=120,
    batch_size=16,
    lr=5e-4,
    aug_profile="mixup_sameclass",
    feature_mode="raw",
    words_file="configs/word78_vocab.txt",
)


def _read_per_class(run: str) -> dict:
    return json.loads((RUNS / run / "reports" / "per_class_accuracy.json").read_text())


def _worst_classes(run: str, n: int) -> list[tuple[str, int, float]]:
    pc = _read_per_class(run)
    items = [(g, v["n"], v["acc"]) for g, v in pc.items() if v.get("acc") is not None]
    items.sort(key=lambda x: x[2])
    return items[:n]


def _train_one(run_name: str, subsample_class: str, pct: float, epochs: int) -> int:
    """Subprocess one training run; returns process return code."""
    history_path = RUNS / run_name / "reports" / "history.json"
    if history_path.exists():
        print(f"    ↩ {run_name} already complete — skipping")
        return 0
    cmd = [
        sys.executable, str(REPO_ROOT / "backend/scripts/train_word_model.py"),
        "--run-name", run_name,
        "--arch", BASE_CONFIG["arch"],
        "--seq-len", str(BASE_CONFIG["seq_len"]),
        "--batch-size", str(BASE_CONFIG["batch_size"]),
        "--lr", str(BASE_CONFIG["lr"]),
        "--aug-profile", BASE_CONFIG["aug_profile"],
        "--feature-mode", BASE_CONFIG["feature_mode"],
        "--words-file", BASE_CONFIG["words_file"],
        "--epochs", str(epochs),
        "--subsample-class", subsample_class,
        "--subsample-pct", str(pct),
    ]
    print(f"    {run_name}")
    t0 = time.time()
    result = subprocess.run(
        cmd, cwd=REPO_ROOT,
        env={**__import__("os").environ, "PYTHONPATH": str(REPO_ROOT)},
        check=False,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    print(f"      → exit={result.returncode}  elapsed={time.time() - t0:.0f}s")
    return result.returncode


def _powerlaw_fit(ns: list[int], accs: list[float]) -> tuple[float, float, float] | None:
    """Fit acc(n) = 1 - a * n^-b via log-linear regression of log(1-acc) vs log(n).

    Returns (a, b, r_squared) or None if not enough finite points or perfect fit.
    """
    pts = [(n, a) for n, a in zip(ns, accs) if 0 <= a < 1 and n > 0]
    if len(pts) < 2:
        return None
    ns_arr = np.array([p[0] for p in pts], dtype=float)
    accs_arr = np.array([p[1] for p in pts], dtype=float)
    err = 1.0 - accs_arr
    err = np.clip(err, 1e-3, 1.0)
    log_n = np.log(ns_arr)
    log_e = np.log(err)
    # log(err) = log(a) - b * log(n)
    slope, intercept = np.polyfit(log_n, log_e, 1)
    b = -slope
    a = math.exp(intercept)
    # R^2
    pred = intercept + slope * log_n
    ss_res = float(np.sum((log_e - pred) ** 2))
    ss_tot = float(np.sum((log_e - log_e.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return a, b, r2


def _predict_n_for_acc(a: float, b: float, target_acc: float) -> float:
    """Invert acc(n) = 1 - a * n^-b → n = (a / (1 - target_acc)) ** (1/b)."""
    err_target = 1.0 - target_acc
    if a <= 0 or b <= 0 or err_target <= 0:
        return math.inf
    return (a / err_target) ** (1.0 / b)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-run", default="word78-seqlen-T=120",
                   help="Run to pick worst classes from.")
    p.add_argument("--worst-n", type=int, default=15)
    p.add_argument("--pcts", nargs="+", type=float, default=[0.25, 0.5, 0.75, 1.0])
    p.add_argument("--epochs", type=int, default=50,
                   help="Reduced from 80 to keep total wall time bounded.")
    p.add_argument("--target-acc", type=float, default=0.9,
                   help="Per-class accuracy target for the data plan.")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    worst = _worst_classes(args.base_run, args.worst_n)
    print(f"Worst {args.worst_n} classes from {args.base_run}:")
    for g, n_test, acc in worst:
        print(f"  {g:15s}  n_test={n_test:3d}  acc={acc:.2f}")

    # Read per-class n_train (from the word audit).
    audit = json.loads((REPORTS / "word_dataset_audit.json").read_text())["per_class"]

    results: dict[str, dict] = {}
    t_start = time.time()
    total = len(worst) * len(args.pcts)
    i = 0
    for gloss, _, base_acc in worst:
        n_train_full = audit.get(gloss, {}).get("n_train", 0)
        results[gloss] = {"n_train_full": n_train_full, "base_acc": base_acc, "points": []}
        for pct in args.pcts:
            i += 1
            run_name = f"lc-{gloss}-pct={int(pct*100):03d}"
            n_kept = max(1, int(round(n_train_full * pct)))
            print(f"\n[{i}/{total}] {gloss} pct={pct:.2f} ({n_kept}/{n_train_full}) → {run_name}")
            if args.dry_run:
                continue
            rc = _train_one(run_name, gloss, pct, args.epochs)
            if rc != 0:
                print(f"  ⚠ failed (rc={rc}), skipping")
                continue
            try:
                pc = _read_per_class(run_name)
                acc = pc.get(gloss, {}).get("acc")
            except Exception as e:
                print(f"  ⚠ could not read result: {e}")
                acc = None
            results[gloss]["points"].append({"pct": pct, "n_kept": n_kept, "acc": acc})

    if args.dry_run:
        print(f"\nDry-run: would launch {total} training runs.")
        return

    print(f"\nFitting power laws (target acc = {args.target_acc:.0%})...")
    plan_rows = []
    for gloss, info in results.items():
        n_full = info["n_train_full"]
        ns = [p["n_kept"] for p in info["points"]]
        accs = [p["acc"] for p in info["points"] if p["acc"] is not None]
        ns = [p["n_kept"] for p in info["points"] if p["acc"] is not None]
        fit = _powerlaw_fit(ns, accs)
        if fit:
            a, b, r2 = fit
            n_target = _predict_n_for_acc(a, b, args.target_acc)
        else:
            a = b = r2 = n_target = None
        plan_rows.append({
            "gloss": gloss, "n_full": n_full,
            "points": info["points"],
            "a": a, "b": b, "r2": r2, "n_target": n_target,
        })

    # Recording-hours calculation. Diversity matrix: ≥3 signers × 3 angles
    # × 2 lighting = 18 takes per sample. Assume one take ~ 6 sec (3 s sign
    # + setup), so 18 × 6 = ~108 s = ~2 min per "sample slot". Conservative:
    # call it 1.5 min per added sample including reset/breaks.
    SEC_PER_SAMPLE = 90.0

    total_extra = 0
    md = [f"# Data Collection Plan v2 (target per-class acc = {args.target_acc:.0%})", "",
          f"Base run: `{args.base_run}` · best config: TCN, seq_len=120, "
          "mixup_sameclass, raw, lr=5e-4, batch=16",
          "",
          "Diversity matrix (mandatory per `record_vocabulary.py --diversity-matrix`):",
          "≥3 signers × 3 angles × 2 lighting = **18 takes per added sample slot**.",
          f"Assumed effort: **{SEC_PER_SAMPLE:.0f} s per added sample** (recording + reset).",
          "",
          "| class | n_train (now) | observed curve | power-law a, b | R² | n needed | n to add | extra hours |",
          "|---|---:|---|---|---:|---:|---:|---:|"]
    for row in plan_rows:
        pts_str = " · ".join(
            f"{p['pct']:.0%}={p['acc']:.2f}" if p['acc'] is not None else f"{p['pct']:.0%}=?"
            for p in row["points"]
        )
        if row["a"] is None:
            md.append(f"| {row['gloss']} | {row['n_full']} | {pts_str} | "
                      f"(no fit) | — | — | — | — |")
            continue
        n_target = row["n_target"]
        if not math.isfinite(n_target):
            n_target_str = "∞"
            n_add = float("inf")
            hrs = float("inf")
        else:
            n_target = max(int(math.ceil(n_target)), row["n_full"])
            n_add = max(0, n_target - row["n_full"])
            hrs = n_add * SEC_PER_SAMPLE / 3600.0
            total_extra += n_add
            n_target_str = f"{n_target}"
        md.append(
            f"| {row['gloss']} | {row['n_full']} | {pts_str} | "
            f"a={row['a']:.2f}, b={row['b']:.2f} | {row['r2']:.2f} | "
            f"{n_target_str} | "
            + (f"{int(n_add)}" if math.isfinite(n_add) else "∞") + " | "
            + (f"{hrs:.1f}" if math.isfinite(hrs) else "—") + " |"
        )

    total_hours = total_extra * SEC_PER_SAMPLE / 3600.0
    md += ["",
           f"**Total extra samples to record:** {total_extra}",
           f"**Total recording time (single-signer-equivalent):** {total_hours:.1f} h",
           f"**Wall time with 3 signers in parallel:** ~{total_hours / 3:.1f} h",
           "",
           "## Raw data points",
           "",
           "```",
           json.dumps(results, indent=2),
           "```",
           "",
           f"Generated in {(time.time() - t_start) / 60:.1f} min.",
           ]
    out = REPORTS / "data_collection_plan_v2.md"
    out.write_text("\n".join(md) + "\n")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
