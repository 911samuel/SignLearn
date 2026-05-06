"""Phase 2 — Subtask 5: Evaluate the trained LSTM on the held-out test split.

Produces:
  artifacts/reports/metrics.json              — accuracy, precision, recall, F1
  artifacts/reports/classification_report.txt — per-class breakdown
  artifacts/reports/confusion_matrix.png      — heatmap

Usage
-----
python scripts/evaluate_model.py
python scripts/evaluate_model.py --model artifacts/checkpoints/lstm_best.keras
python scripts/evaluate_model.py --model artifacts/checkpoints/lstm_final.h5
"""

import argparse
import json
import sys
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

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from backend.data.dataset import build_dataset
from backend.model.config import (
    CHECKPOINTS_DIR,
    PROCESSED_DIR,
    REPORTS_DIR,
    compact_class_names,
    compact_label_map,
)
from scripts.train_model import _remap_labels


def evaluate(model_path: Path, data_dir: Path, reports_dir: Path) -> dict:
    """Load model, run on test split, write report files, return metrics dict."""
    model = tf.keras.models.load_model(str(model_path))
    print(f"Loaded model: {model_path}")

    cmap  = compact_label_map(processed_dir=data_dir)
    names = compact_class_names(processed_dir=data_dir)

    test_ds = build_dataset("test", batch_size=64, augment=False, processed_dir=data_dir)
    test_ds = _remap_labels(test_ds, cmap)

    y_true, y_pred = [], []
    for seqs, labels in test_ds:
        preds = model.predict(seqs, verbose=0)
        y_true.extend(labels.numpy().tolist())
        y_pred.extend(np.argmax(preds, axis=1).tolist())

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

    metrics = {
        "model": rel_model,
        "test_samples": int(len(y_true)),
        "num_classes": len(names),
        "class_names": names,
        "accuracy": round(acc, 4),
        "precision_macro": round(precision, 4),
        "recall_macro": round(recall, 4),
        "f1_macro": round(f1, 4),
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


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Evaluate SignLearn LSTM on test split")
    p.add_argument(
        "--model",
        type=Path,
        default=CHECKPOINTS_DIR / "lstm_best.keras",
        help="Path to saved model (.h5 or .keras)",
    )
    p.add_argument("--data-dir",    type=Path, default=PROCESSED_DIR)
    p.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    p.add_argument(
        "--min-accuracy",
        type=float,
        default=0.85,
        help="Phase 2 target accuracy. Exit non-zero if test accuracy is below this. "
             "Set to 0 to disable the gate.",
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    metrics = evaluate(args.model, data_dir=args.data_dir, reports_dir=args.reports_dir)
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
    print(f"\n[PASS] Test accuracy meets Phase 2 target (>= {args.min_accuracy:.1%}).")
