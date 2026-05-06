"""Phase 2 — Subtask 3: Train the stacked LSTM on present ASL classes.

Usage
-----
python scripts/train_model.py                         # default config
python scripts/train_model.py --epochs 50 --batch-size 16
python scripts/train_model.py --data-dir data/processed --out-dir artifacts
"""

import argparse
import json
import sys
from pathlib import Path

import tensorflow as tf

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from backend.data.dataset import build_dataset
from backend.model.architecture import build_lstm
from backend.model.config import (
    ARTIFACTS_DIR,
    CHECKPOINTS_DIR,
    LOGS_DIR,
    PROCESSED_DIR,
    REPORTS_DIR,
    TrainConfig,
    compact_label_map,
)


def _remap_labels(ds: tf.data.Dataset, cmap: dict[int, int]) -> tf.data.Dataset:
    """Remap full-vocab label indices to compact 0..N-1 indices.

    The dataset pipeline returns labels as full vocabulary indices
    (e.g. zero=26, one=27, …nine=35). The model output layer has
    num_classes=10, so we remap to 0..9 here.
    """
    lookup = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            keys=list(cmap.keys()),
            values=list(cmap.values()),
            key_dtype=tf.int32,
            value_dtype=tf.int32,
        ),
        default_value=-1,
    )

    def _map(seq, label):
        new_label = lookup.lookup(label)
        # Fail loudly if a split contains a label not present in the compact map.
        # This catches the case where val/test has classes absent from train,
        # which would otherwise silently feed -1 into the loss.
        tf.debugging.assert_greater_equal(
            new_label, 0,
            message="_remap_labels: encountered label not in compact map",
        )
        return seq, new_label

    return ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE)


def train(config: TrainConfig, data_dir: Path, out_dir: Path) -> dict:
    """Run training and return the history dict."""
    tf.random.set_seed(config.seed)

    cmap = compact_label_map(processed_dir=data_dir)

    train_ds = build_dataset("train", batch_size=config.batch_size, augment=True, processed_dir=data_dir)
    val_ds   = build_dataset("val",   batch_size=config.batch_size, augment=False, processed_dir=data_dir)

    train_ds = _remap_labels(train_ds, cmap)
    val_ds   = _remap_labels(val_ds,   cmap)

    model = build_lstm(config)
    model.summary()

    checkpoints_dir = out_dir / "checkpoints"
    logs_dir        = out_dir / "logs"
    reports_dir     = out_dir / "reports"
    for d in (checkpoints_dir, logs_dir, reports_dir):
        d.mkdir(parents=True, exist_ok=True)

    best_ckpt   = checkpoints_dir / "lstm_best.keras"
    final_keras = checkpoints_dir / "lstm_final.keras"

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(best_ckpt),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=config.early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=config.reduce_lr_factor,
            patience=config.reduce_lr_patience,
            min_lr=config.min_lr,
            verbose=1,
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=str(logs_dir),
            histogram_freq=1,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    model.save(str(final_keras))
    print(f"\nFinal model saved → {final_keras}")

    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    history_path = reports_dir / "history.json"
    with open(history_path, "w") as f:
        json.dump(history_dict, f, indent=2)
    print(f"Training history  → {history_path}")

    best_val_acc = max(history_dict.get("val_accuracy", [0]))
    best_epoch   = history_dict.get("val_accuracy", [0]).index(best_val_acc) + 1
    print(f"\nBest val_accuracy: {best_val_acc:.4f} at epoch {best_epoch}")

    return history_dict


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train SignLearn LSTM (Phase 2)")
    p.add_argument("--epochs",      type=int,  default=None, help="Override TrainConfig.epochs")
    p.add_argument("--batch-size",  type=int,  default=None, help="Override TrainConfig.batch_size")
    p.add_argument("--lr",          type=float,default=None, help="Override learning rate")
    p.add_argument("--data-dir",    type=Path, default=PROCESSED_DIR)
    p.add_argument("--out-dir",     type=Path, default=ARTIFACTS_DIR)
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()

    config = TrainConfig()
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.lr is not None:
        config.learning_rate = args.lr

    train(config, data_dir=args.data_dir, out_dir=args.out_dir)
