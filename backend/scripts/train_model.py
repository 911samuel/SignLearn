"""Phase 2/5 — Train a SignLearn sequence classifier.

Supports multiple architectures via :data:`backend.model.architectures.ARCHITECTURE_REGISTRY`.

Usage
-----
python backend/scripts/train_model.py                                       # baseline LSTM
python backend/scripts/train_model.py --arch bilstm --run-name bilstm-v1
python backend/scripts/train_model.py --arch transformer --run-name tx-v1 --epochs 80
python backend/scripts/train_model.py --feature-mode raw+velocity --run-name lstm-vel
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import tensorflow as tf

_REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from backend.data.augment import TRAINING_PROBS
from backend.data.dataset import build_dataset
from backend.data.features import output_dim
from backend.data.label_map import load_label_map
from backend.model.architectures import build as build_arch
from backend.model.config import (
    ARTIFACTS_DIR,
    CHECKPOINTS_DIR,
    FEATURE_DIM,
    LOGS_DIR,
    PROCESSED_DIR,
    REPORTS_DIR,
    SEQUENCE_LEN,
    TrainConfig,
    compact_label_map,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
_log = logging.getLogger(__name__)


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
        tf.debugging.assert_greater_equal(
            new_label, 0,
            message="_remap_labels: encountered label not in compact map",
        )
        return seq, new_label

    return ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE)


def _run_dirs(out_dir: Path, run_name: str | None) -> tuple[Path, Path, Path]:
    """Return (checkpoints, logs, reports) directories for this run.

    Without ``run_name`` the layout matches the legacy Phase 2 expectation:
    ``<out_dir>/{checkpoints,logs,reports}``. With a ``run_name`` everything
    is nested under ``<out_dir>/runs/<run_name>/`` so multiple experiments can
    coexist.
    """
    if run_name:
        root = out_dir / "runs" / run_name
    else:
        root = out_dir
    ck = root / "checkpoints"
    lg = root / "logs"
    rp = root / "reports"
    for d in (ck, lg, rp):
        d.mkdir(parents=True, exist_ok=True)
    return ck, lg, rp


def train(
    config: TrainConfig,
    data_dir: Path,
    out_dir: Path,
    run_name: str | None = None,
) -> dict:
    """Run training and return the history dict.

    The architecture is selected by ``config.arch_name``. Checkpoints are saved
    as ``<arch>_best.keras`` and ``<arch>_final.keras`` so multiple architectures
    can share an output directory without clobbering each other.
    """
    tf.random.set_seed(config.seed)

    cmap = compact_label_map(processed_dir=data_dir)
    config.num_classes = len(cmap)
    # Adjust the model input width if engineered features are enabled.
    config.input_shape = (SEQUENCE_LEN, output_dim(config.feature_mode))

    full_vocab_size = len(load_label_map())
    if len(cmap) < full_vocab_size:
        missing = full_vocab_size - len(cmap)
        _log.warning(
            "Training on %d/%d classes — %d classes have no data in %s. "
            "Run extract_landmarks.py to collect samples for the remaining classes.",
            len(cmap), full_vocab_size, missing, data_dir,
        )

    train_ds = build_dataset(
        "train", batch_size=config.batch_size, augment=True,
        processed_dir=data_dir, feature_mode=config.feature_mode,
        augment_probs=TRAINING_PROBS,
    )
    val_ds = build_dataset(
        "val", batch_size=config.batch_size, augment=False,
        processed_dir=data_dir, feature_mode=config.feature_mode,
    )

    train_ds = _remap_labels(train_ds, cmap)
    val_ds   = _remap_labels(val_ds,   cmap)

    model = build_arch(config.arch_name, config)
    model.summary(print_fn=_log.info)

    checkpoints_dir, logs_dir, reports_dir = _run_dirs(out_dir, run_name)
    arch = config.arch_name
    best_ckpt   = checkpoints_dir / f"{arch}_best.keras"
    final_keras = checkpoints_dir / f"{arch}_final.keras"

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

    t0 = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.epochs,
        callbacks=callbacks,
        verbose=1,
    )
    elapsed = time.time() - t0

    model.save(str(final_keras))
    print(f"\nFinal model saved → {final_keras}")

    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    history_path = reports_dir / "history.json"
    with open(history_path, "w") as f:
        json.dump(history_dict, f, indent=2)
    print(f"Training history  → {history_path}")

    # Persist run metadata so evaluate_model.py can rehydrate the same config.
    meta = {
        "arch_name":       config.arch_name,
        "feature_mode":    config.feature_mode,
        "num_classes":     config.num_classes,
        "input_shape":     list(config.input_shape),
        "param_count":     int(model.count_params()),
        "batch_size":      config.batch_size,
        "epochs_run":      len(history_dict.get("loss", [])),
        "elapsed_seconds": round(elapsed, 2),
        "best_val_acc":    max(history_dict.get("val_accuracy", [0.0])),
        "best_val_loss":   min(history_dict.get("val_loss", [float("inf")])),
        "seed":            config.seed,
        "run_name":        run_name,
        "best_checkpoint": str(best_ckpt),
        "final_checkpoint": str(final_keras),
    }
    config_path = reports_dir / "config.json"
    config_path.write_text(json.dumps(meta, indent=2))
    print(f"Run metadata      → {config_path}")

    best_val_acc = max(history_dict.get("val_accuracy", [0]))
    best_epoch   = history_dict.get("val_accuracy", [0]).index(best_val_acc) + 1
    print(f"\nBest val_accuracy: {best_val_acc:.4f} at epoch {best_epoch} "
          f"({elapsed:.1f}s total)")

    return history_dict


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train a SignLearn sequence classifier")
    p.add_argument("--arch",         type=str,  default="lstm",
                   choices=["lstm", "bilstm", "transformer"])
    p.add_argument("--feature-mode", type=str,  default="raw",
                   choices=["raw", "raw+velocity", "raw+velocity+angles"])
    p.add_argument("--run-name",     type=str,  default=None,
                   help="Optional subdirectory under <out-dir>/runs/ for this experiment")
    p.add_argument("--epochs",       type=int,  default=None)
    p.add_argument("--batch-size",   type=int,  default=None)
    p.add_argument("--lr",           type=float,default=None)
    p.add_argument("--data-dir",     type=Path, default=PROCESSED_DIR)
    p.add_argument("--out-dir",      type=Path, default=ARTIFACTS_DIR)
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()

    config = TrainConfig(arch_name=args.arch, feature_mode=args.feature_mode)
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.lr is not None:
        config.learning_rate = args.lr

    train(config, data_dir=args.data_dir, out_dir=args.out_dir, run_name=args.run_name)
