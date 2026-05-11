"""Phase 2 — Hyperparameter grid search for the SignLearn LSTM.

Performs an exhaustive (or random-sampled) grid search over the key
hyperparameters, training each configuration for a fixed number of epochs and
recording val_accuracy. Results are written to artifacts/reports/hp_search.json
and a Markdown summary to artifacts/reports/hp_search.md.

No extra libraries required — uses only TensorFlow/Keras and stdlib.

Usage
-----
# Full grid search (all combinations of the default grid)
python backend/scripts/tune_hyperparams.py

# Random search — sample 10 configs at random from the full grid
python backend/scripts/tune_hyperparams.py --random 10

# Quick smoke test — 1 epoch, fewer configs
python backend/scripts/tune_hyperparams.py --epochs 1 --random 3

# Custom data dir
python backend/scripts/tune_hyperparams.py --data-dir tests/fixtures/processed_mini

# Use GPU (default: cpu)
python backend/scripts/tune_hyperparams.py --device gpu
"""

import argparse
import json
import os
import random
import sys
import time
from dataclasses import asdict
from itertools import product
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from backend.data.dataset import build_dataset
from backend.model.architecture import build_lstm
from backend.model.config import (
    ARTIFACTS_DIR,
    PROCESSED_DIR,
    REPORTS_DIR,
    TrainConfig,
    compact_label_map,
)
from backend.scripts.train_model import _remap_labels

# ---------------------------------------------------------------------------
# Default hyperparameter grid
# ---------------------------------------------------------------------------

_DEFAULT_GRID: dict[str, list] = {
    "lstm_units":    [(128, 64), (256, 128), (64, 32)],
    "dropout":       [0.3, 0.4, 0.5],
    "learning_rate": [1e-3, 5e-4, 1e-4],
}


def _grid_configs(grid: dict[str, list]) -> list[dict]:
    """Enumerate all combinations from a hyperparameter grid dict."""
    keys = list(grid.keys())
    values = list(grid.values())
    return [dict(zip(keys, combo)) for combo in product(*values)]


def _train_one(
    hparams: dict,
    *,
    data_dir: Path,
    epochs: int,
    batch_size: int,
    run_id: int,
) -> dict:
    """Train one config for *epochs* epochs and return a result dict."""
    import tensorflow as tf

    tf.random.set_seed(42)

    cmap = compact_label_map(processed_dir=data_dir)
    num_classes = len(cmap)

    config = TrainConfig(
        num_classes=num_classes,
        epochs=epochs,
        batch_size=batch_size,
        lstm_units=hparams["lstm_units"],
        dropout=hparams["dropout"],
        learning_rate=hparams["learning_rate"],
        # Disable early stopping / LR decay during search so epoch counts
        # are consistent across all configurations.
        early_stopping_patience=epochs + 1,
        reduce_lr_patience=epochs + 1,
    )

    train_ds = build_dataset("train", batch_size=batch_size, augment=True, processed_dir=data_dir)
    val_ds   = build_dataset("val",   batch_size=batch_size, augment=False, processed_dir=data_dir)
    train_ds = _remap_labels(train_ds, cmap)
    val_ds   = _remap_labels(val_ds,   cmap)

    model = build_lstm(config)

    t0 = time.perf_counter()
    hist = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        verbose=0,
    )
    elapsed = time.perf_counter() - t0

    val_accs = hist.history.get("val_accuracy", [0.0])
    best_val_acc = float(max(val_accs))
    best_epoch   = int(val_accs.index(best_val_acc)) + 1

    return {
        "run_id": run_id,
        "hparams": {
            "lstm_units": list(hparams["lstm_units"]),
            "dropout": hparams["dropout"],
            "learning_rate": hparams["learning_rate"],
        },
        "best_val_accuracy": round(best_val_acc, 4),
        "best_epoch": best_epoch,
        "epochs_trained": epochs,
        "training_time_s": round(elapsed, 2),
        "num_params": int(model.count_params()),
    }


def run_search(
    grid: dict[str, list],
    *,
    data_dir: Path,
    epochs: int,
    batch_size: int,
    n_random: int | None,
    reports_dir: Path,
) -> list[dict]:
    """Run the full or random-sampled grid search. Returns sorted results."""
    all_configs = _grid_configs(grid)
    if n_random is not None:
        rng = random.Random(42)
        all_configs = rng.sample(all_configs, min(n_random, len(all_configs)))

    total = len(all_configs)
    print(f"\nHyperparameter search: {total} configuration(s), {epochs} epoch(s) each")
    print(f"Data dir: {data_dir}")
    print(f"{'─' * 60}")

    results: list[dict] = []
    for i, hparams in enumerate(all_configs, 1):
        desc = (
            f"LSTM{hparams['lstm_units']}  "
            f"drop={hparams['dropout']}  "
            f"lr={hparams['learning_rate']:.0e}"
        )
        print(f"\n[{i:2d}/{total}] {desc}")
        try:
            result = _train_one(
                hparams,
                data_dir=data_dir,
                epochs=epochs,
                batch_size=batch_size,
                run_id=i,
            )
            results.append(result)
            print(
                f"        → val_acc={result['best_val_accuracy']:.4f}  "
                f"(epoch {result['best_epoch']})  "
                f"time={result['training_time_s']:.1f}s  "
                f"params={result['num_params']:,}"
            )
        except Exception as exc:
            print(f"        → FAILED: {exc}")
            results.append({"run_id": i, "hparams": hparams, "error": str(exc)})

    # Sort by best val_accuracy descending; errored runs last
    results.sort(key=lambda r: r.get("best_val_accuracy", -1), reverse=True)
    return results


def write_reports(results: list[dict], reports_dir: Path, grid: dict) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    json_path = reports_dir / "hp_search.json"
    with open(json_path, "w") as f:
        json.dump({"grid": {k: [list(v) if isinstance(v, tuple) else v for v in vals]
                              for k, vals in grid.items()},
                   "results": results}, f, indent=2)
    print(f"\nSearch results → {json_path}")

    # Markdown summary
    best = next((r for r in results if "best_val_accuracy" in r), None)
    md_lines = [
        "# Phase 2 — Hyperparameter Search Results\n",
        f"**Configurations tested:** {len(results)}  ",
    ]
    if best:
        md_lines.append(
            f"**Best val_accuracy:** {best['best_val_accuracy']:.4f}  \n"
            f"**Best config:** `{best['hparams']}`\n"
        )

    md_lines += [
        "## Results (sorted by val_accuracy)\n",
        "| Run | lstm_units | dropout | lr | val_acc | epoch | params | time (s) |",
        "|-----|-----------|---------|-----|---------|-------|--------|---------|",
    ]
    for r in results:
        if "error" in r:
            md_lines.append(
                f"| {r['run_id']} | — | — | — | ERROR | — | — | — |"
            )
            continue
        hp = r["hparams"]
        md_lines.append(
            f"| {r['run_id']} "
            f"| {hp['lstm_units']} "
            f"| {hp['dropout']} "
            f"| {hp['learning_rate']:.0e} "
            f"| {r['best_val_accuracy']:.4f} "
            f"| {r['best_epoch']} "
            f"| {r['num_params']:,} "
            f"| {r['training_time_s']:.1f} |"
        )

    md_path = reports_dir / "hp_search.md"
    md_path.write_text("\n".join(md_lines) + "\n")
    print(f"Markdown report → {md_path}")


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Grid / random hyperparameter search for SignLearn LSTM (Phase 2)"
    )
    p.add_argument(
        "--data-dir", type=Path, default=PROCESSED_DIR,
        help="Processed dataset root (default: data/processed)",
    )
    p.add_argument(
        "--epochs", type=int, default=10,
        help="Training epochs per config (default: 10). Use 1 for smoke tests.",
    )
    p.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size (default: 32)",
    )
    p.add_argument(
        "--random", type=int, default=None, metavar="N",
        help="Random-sample N configs from the full grid instead of running all.",
    )
    p.add_argument(
        "--reports-dir", type=Path, default=REPORTS_DIR,
        help="Output directory for reports (default: artifacts/reports)",
    )
    p.add_argument(
        "--device", choices=["cpu", "gpu"], default="cpu",
        help="'cpu' forces CPU-only (default); 'gpu' allows GPU/MPS.",
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()

    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    results = run_search(
        _DEFAULT_GRID,
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        n_random=args.random,
        reports_dir=args.reports_dir,
    )

    write_reports(results, args.reports_dir, _DEFAULT_GRID)

    print(f"\n{'─' * 60}")
    best = next((r for r in results if "best_val_accuracy" in r), None)
    if best:
        print(f"Best configuration:")
        print(f"  lstm_units    : {best['hparams']['lstm_units']}")
        print(f"  dropout       : {best['hparams']['dropout']}")
        print(f"  learning_rate : {best['hparams']['learning_rate']}")
        print(f"  val_accuracy  : {best['best_val_accuracy']:.4f}")
        print(f"\nUpdate TrainConfig in backend/model/config.py with these values,")
        print(f"then run: python backend/scripts/train_model.py")
    print(f"{'─' * 60}\n")
