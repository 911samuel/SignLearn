"""Evaluate a trained SignLearn model on the held-out test split.

Two modes:

1. **Single-run** (legacy / Phase 2):
   ``python backend/scripts/evaluate_model.py --model artifacts/checkpoints/lstm_best.keras``

   Produces in ``--reports-dir``:
     - metrics.json
     - classification_report.txt
     - confusion_matrix.png

2. **Multi-run comparison** (Phase 6):
   ``python backend/scripts/evaluate_model.py --runs lstm-v1 bilstm-v1 tx-v1``

   For each run located at ``<artifacts>/runs/<name>/`` it evaluates the
   ``*_best.keras`` checkpoint and aggregates a side-by-side report at
   ``artifacts/reports/model_comparison.md``.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

_REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from backend.data.dataset import build_dataset
from backend.model.config import (
    ARTIFACTS_DIR,
    CHECKPOINTS_DIR,
    PROCESSED_DIR,
    REPORTS_DIR,
    compact_class_names,
    compact_label_map,
)
from backend.scripts.train_model import _remap_labels


def evaluate(
    model_path: Path,
    data_dir: Path,
    reports_dir: Path,
    feature_mode: str = "raw",
) -> dict:
    """Load model, run on test split, write report files, return metrics dict."""
    model = tf.keras.models.load_model(str(model_path))
    print(f"Loaded model: {model_path}")

    cmap  = compact_label_map(processed_dir=data_dir)
    names = compact_class_names(processed_dir=data_dir)

    test_ds = build_dataset(
        "test", batch_size=64, augment=False,
        processed_dir=data_dir, feature_mode=feature_mode,
    )
    test_ds = _remap_labels(test_ds, cmap)

    y_true, y_pred = [], []
    t0 = time.time()
    for seqs, labels in test_ds:
        preds = model.predict(seqs, verbose=0)
        y_true.extend(labels.numpy().tolist())
        y_pred.extend(np.argmax(preds, axis=1).tolist())
    inference_seconds = time.time() - t0

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc       = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    recall    = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1        = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    try:
        rel_model = str(Path(model_path).resolve().relative_to(_REPO_ROOT))
    except ValueError:
        rel_model = str(model_path)

    # Per-class precision/recall/f1 for easy programmatic access
    from sklearn.metrics import precision_recall_fscore_support
    per_class_p, per_class_r, per_class_f, per_class_support = (
        precision_recall_fscore_support(y_true, y_pred, labels=list(range(len(names))), zero_division=0)
    )
    per_class_metrics = {
        names[i]: {
            "precision": round(float(per_class_p[i]), 4),
            "recall":    round(float(per_class_r[i]), 4),
            "f1":        round(float(per_class_f[i]), 4),
            "support":   int(per_class_support[i]),
        }
        for i in range(len(names))
    }

    # Top confusion pairs (true_label, predicted_label, count)
    import numpy as np
    cm_tmp = confusion_matrix(y_true, y_pred, labels=list(range(len(names))))
    np.fill_diagonal(cm_tmp, 0)
    top_confusion_pairs = []
    while len(top_confusion_pairs) < 20:
        i, j = np.unravel_index(np.argmax(cm_tmp), cm_tmp.shape)
        cnt = int(cm_tmp[i, j])
        if cnt == 0:
            break
        top_confusion_pairs.append({"true": names[i], "predicted": names[j], "count": cnt})
        cm_tmp[i, j] = 0

    metrics = {
        "model": rel_model,
        "test_samples": int(len(y_true)),
        "num_classes": len(names),
        "class_names": names,
        "accuracy": round(acc, 4),
        "precision_macro": round(precision, 4),
        "recall_macro": round(recall, 4),
        "f1_macro": round(f1, 4),
        "inference_seconds_total": round(inference_seconds, 3),
        "inference_seconds_per_sample": round(inference_seconds / max(1, len(y_true)), 5),
        "param_count": int(model.count_params()),
        "feature_mode": feature_mode,
        "per_class": per_class_metrics,
        "top_confusion_pairs": top_confusion_pairs,
    }

    reports_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = reports_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics          → {metrics_path}")

    report_str = classification_report(y_true, y_pred, target_names=names, zero_division=0)
    report_path = reports_dir / "classification_report.txt"
    report_path.write_text(report_str)
    print(f"Class report     → {report_path}")
    print("\n" + report_str)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(names))))
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=names,
        yticklabels=names,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(f"Confusion Matrix — Test Set  (acc={acc:.1%})", fontsize=13)
    plt.tight_layout()
    cm_path = reports_dir / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix → {cm_path}")

    return metrics


# ---------------------------------------------------------------------------
# Phase 6 — multi-run comparison
# ---------------------------------------------------------------------------

def _load_run_meta(run_dir: Path) -> dict:
    cfg = run_dir / "reports" / "config.json"
    if not cfg.exists():
        return {}
    return json.loads(cfg.read_text())


def _pick_checkpoint(run_dir: Path, meta: dict) -> Path:
    """Prefer the explicit best-checkpoint recorded in config.json, else search."""
    if meta.get("best_checkpoint") and Path(meta["best_checkpoint"]).exists():
        return Path(meta["best_checkpoint"])
    # Fallback: glob the run's checkpoints dir.
    ck_dir = run_dir / "checkpoints"
    bests = sorted(ck_dir.glob("*_best.keras"))
    if bests:
        return bests[0]
    finals = sorted(ck_dir.glob("*_final.keras"))
    if finals:
        return finals[0]
    raise FileNotFoundError(f"No checkpoints found under {ck_dir}")


def compare_runs(
    run_names: list[str],
    runs_root: Path,
    data_dir: Path,
    out_path: Path,
) -> dict:
    """Evaluate each run and write a side-by-side comparison report."""
    rows = []
    for name in run_names:
        run_dir = runs_root / name
        if not run_dir.exists():
            print(f"[skip] {name}: directory not found ({run_dir})", file=sys.stderr)
            continue
        meta = _load_run_meta(run_dir)
        try:
            ckpt = _pick_checkpoint(run_dir, meta)
        except FileNotFoundError as e:
            print(f"[skip] {name}: {e}", file=sys.stderr)
            continue
        feat = meta.get("feature_mode", "raw")
        reports = run_dir / "reports"
        metrics = evaluate(ckpt, data_dir=data_dir, reports_dir=reports, feature_mode=feat)
        rows.append({
            "run":          name,
            "arch":         meta.get("arch_name", "?"),
            "feature_mode": feat,
            "params":       metrics["param_count"],
            "epochs_run":   meta.get("epochs_run"),
            "train_sec":    meta.get("elapsed_seconds"),
            "accuracy":     metrics["accuracy"],
            "f1_macro":     metrics["f1_macro"],
            "precision":    metrics["precision_macro"],
            "recall":       metrics["recall_macro"],
            "best_val_acc": meta.get("best_val_acc"),
            "ms_per_sample": round(metrics["inference_seconds_per_sample"] * 1000, 3),
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# SignLearn — Model Comparison", "",
             "| Run | Arch | Features | Params | Epochs | Train (s) | Best val | Test acc | F1 | Prec | Rec | ms/sample |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(
            f"| `{r['run']}` | {r['arch']} | {r['feature_mode']} | "
            f"{r['params']:,} | {r['epochs_run']} | {r['train_sec']} | "
            f"{(r['best_val_acc'] or 0):.4f} | **{r['accuracy']:.4f}** | "
            f"{r['f1_macro']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} | "
            f"{r['ms_per_sample']} |"
        )

    if rows:
        best = max(rows, key=lambda r: r["accuracy"])
        lines += ["", f"**Top run:** `{best['run']}` — "
                  f"test accuracy **{best['accuracy']:.1%}**, "
                  f"F1 {best['f1_macro']:.1%}, "
                  f"{best['ms_per_sample']} ms/sample."]
        passes = [r for r in rows if r["accuracy"] >= 0.85]
        if passes:
            lines.append(f"\n✅ {len(passes)} run(s) meet the 85% Phase 2 target.")
        else:
            lines.append("\n⚠️ No run met the 85% Phase 2 target.")

        # Top confusion pairs from the best run's metrics
        best_metrics_path = runs_root / best["run"] / "reports" / "metrics.json"
        if best_metrics_path.exists():
            import json as _json
            bm = _json.load(open(best_metrics_path))
            pairs = bm.get("top_confusion_pairs", [])
            if pairs:
                lines += ["", f"### Top confusion pairs (best run: `{best['run']}`)", "",
                          "| True | → Predicted | Count |", "|---|---|---:|"]
                for p in pairs[:10]:
                    lines.append(f"| `{p['true']}` | `{p['predicted']}` | {p['count']} |")

        # Per-class failures (F1 < 0.5) from best run
        if "per_class" in bm:
            low_f1 = [(cls, d["f1"], d["support"])
                      for cls, d in bm["per_class"].items() if d["f1"] < 0.5]
            if low_f1:
                low_f1.sort(key=lambda x: x[1])
                lines += ["", "### Classes with F1 < 0.50 (best run)", "",
                          "| Class | F1 | Support |", "|---|---:|---:|"]
                for cls, f1, sup in low_f1:
                    lines.append(f"| `{cls}` | {f1:.3f} | {sup} |")
    else:
        lines.append("\nNo runs evaluated.")

    out_path.write_text("\n".join(lines))
    print(f"\nComparison report → {out_path}")
    return {"rows": rows, "report_path": str(out_path)}


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Evaluate SignLearn models on the test split")
    p.add_argument("--model", type=Path,
                   default=CHECKPOINTS_DIR / "lstm_best.keras",
                   help="Path to saved model (.h5 or .keras) — single-run mode")
    p.add_argument("--runs", nargs="+", default=None,
                   help="Run names to compare (under <artifacts>/runs/<name>)")
    p.add_argument("--feature-mode", type=str, default="raw",
                   choices=["raw", "raw+velocity", "raw+velocity+angles",
                            "raw+velocity+acceleration", "engineered"],
                   help="Feature mode for single-run evaluation (multi-run reads each run's config.json)")
    p.add_argument("--runs-root", type=Path, default=ARTIFACTS_DIR / "runs")
    p.add_argument("--data-dir", type=Path, default=PROCESSED_DIR)
    p.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    p.add_argument("--comparison-out", type=Path,
                   default=REPORTS_DIR / "model_comparison.md")
    p.add_argument("--min-accuracy", type=float, default=0.85,
                   help="Target accuracy. Single-run mode exits non-zero if below this. "
                        "Set to 0 to disable the gate.")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()

    if args.runs:
        result = compare_runs(
            args.runs,
            runs_root=args.runs_root,
            data_dir=args.data_dir,
            out_path=args.comparison_out,
        )
        if args.min_accuracy > 0:
            best = max((r["accuracy"] for r in result["rows"]), default=0.0)
            if best < args.min_accuracy:
                print(f"\n[FAIL] Best test accuracy {best:.1%} below "
                      f"{args.min_accuracy:.1%}.", file=sys.stderr)
                sys.exit(1)
            print(f"\n[PASS] Best run meets the {args.min_accuracy:.1%} target.")
        sys.exit(0)

    metrics = evaluate(
        args.model,
        data_dir=args.data_dir,
        reports_dir=args.reports_dir,
        feature_mode=args.feature_mode,
    )
    print(f"\nSummary: acc={metrics['accuracy']:.1%}  "
          f"F1={metrics['f1_macro']:.1%}  "
          f"precision={metrics['precision_macro']:.1%}  "
          f"recall={metrics['recall_macro']:.1%}")

    if args.min_accuracy > 0 and metrics["accuracy"] < args.min_accuracy:
        print(
            f"\n[FAIL] Test accuracy {metrics['accuracy']:.1%} is below the "
            f"Phase 2 target of {args.min_accuracy:.1%}.",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"\n[PASS] Test accuracy meets target (>= {args.min_accuracy:.1%}).")
