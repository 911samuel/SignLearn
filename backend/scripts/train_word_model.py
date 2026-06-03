"""
Train the word-level sign classifier on data/processed/words/.

Standalone trainer — does NOT use the letter-model label map (which is locked
to ``docs/vocabulary.md``). Instead it derives classes from the curated word
list at ``configs/wlasl_words_curated.txt``, so WLASL glosses outside the
vocabulary file (``africa``, ``computer``, …) still train.

Inputs (written by extract_wlasl_landmarks.py / extract_youtube_signs.py):
  data/processed/words/{train,val,test}/<gloss>_s*_<idx>.npy   shape (T, 126)

Output:
  artifacts/runs/<run-name>/
    checkpoints/<arch>_best.keras
    reports/{config.json, history.json, classification_report.txt}
    word_label_map.json

Usage:
  python backend/scripts/train_word_model.py --arch bilstm --run-name word-bilstm-v1
  python backend/scripts/train_word_model.py --arch tcn --seq-len 80 --epochs 60
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from backend.data.constants import FEATURE_DIM
from backend.model.architectures import build as build_arch
from backend.model.config import TrainConfig

DATA_ROOT = REPO_ROOT / "data" / "processed" / "words"
CURATED_PATH = REPO_ROOT / "configs" / "wlasl_words_curated.txt"
ARTIFACTS = REPO_ROOT / "artifacts" / "runs"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def load_label_map(curated_path: Path) -> dict[str, int]:
    glosses = sorted({
        w.strip() for w in curated_path.read_text().splitlines()
        if w.strip() and not w.startswith("#")
    })
    return {g: i for i, g in enumerate(glosses)}


def _gloss_from_filename(path: Path) -> str:
    """``hello_s05_0001.npy`` → ``hello``;  ``thank_you_syt0a3f_0002.npy`` → ``thank_you``."""
    stem = path.stem
    # Strip the trailing _<idx>_<sample> = 2 trailing tokens after gloss.
    parts = stem.split("_")
    # Identify where the signer token begins (starts with 's', followed by digits or 'yt').
    for i, p in enumerate(parts):
        if p.startswith("s") and len(p) > 1 and (p[1:].isdigit() or p.startswith("syt")):
            return "_".join(parts[:i])
    # Fallback: drop the last 2 tokens.
    return "_".join(parts[:-2]) if len(parts) >= 3 else stem


def list_split(split: str, label_map: dict[str, int], seq_len: int) -> list[tuple[Path, int]]:
    split_dir = DATA_ROOT / split
    items: list[tuple[Path, int]] = []
    skipped_unknown = 0
    skipped_shape = 0
    for npy in split_dir.glob("*.npy"):
        gloss = _gloss_from_filename(npy)
        if gloss not in label_map:
            skipped_unknown += 1
            continue
        arr = np.load(npy, mmap_mode="r")
        if arr.shape != (seq_len, FEATURE_DIM):
            skipped_shape += 1
            continue
        items.append((npy, label_map[gloss]))
    if skipped_unknown:
        log.warning("%s: skipped %d file(s) with gloss not in label_map", split, skipped_unknown)
    if skipped_shape:
        log.warning("%s: skipped %d file(s) with wrong shape (expected (%d, %d))",
                    split, skipped_shape, seq_len, FEATURE_DIM)
    return items


def build_dataset(
    items: list[tuple[Path, int]],
    seq_len: int,
    batch_size: int,
    shuffle: bool,
) -> tf.data.Dataset:
    paths = [str(p) for p, _ in items]
    labels = [int(y) for _, y in items]

    def _load(path: tf.Tensor, label: tf.Tensor):
        def _np_load(p):
            return np.load(p.decode()).astype(np.float32)
        seq = tf.numpy_function(_np_load, [path], tf.float32)
        seq.set_shape([seq_len, FEATURE_DIM])
        return seq, label

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(items), 2048), reshuffle_each_iteration=True)
    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args) -> None:
    label_map = load_label_map(CURATED_PATH)
    n_classes = len(label_map)
    log.info("Classes: %d", n_classes)

    train_items = list_split("train", label_map, args.seq_len)
    val_items = list_split("val", label_map, args.seq_len)
    test_items = list_split("test", label_map, args.seq_len)
    log.info("Items — train=%d  val=%d  test=%d", len(train_items), len(val_items), len(test_items))

    if not train_items:
        sys.exit("No training data found. Run extract_wlasl_landmarks.py first.")

    # Class-balance check.
    from collections import Counter
    counts = Counter(y for _, y in train_items)
    inv = {v: k for k, v in label_map.items()}
    weak = [(inv[c], n) for c, n in counts.items() if n < 5]
    if weak:
        log.warning("Classes with <5 train samples: %s", weak)

    train_ds = build_dataset(train_items, args.seq_len, args.batch_size, shuffle=True)
    val_ds = build_dataset(val_items, args.seq_len, args.batch_size, shuffle=False) if val_items else None
    test_ds = build_dataset(test_items, args.seq_len, args.batch_size, shuffle=False) if test_items else None

    cfg = TrainConfig(
        arch_name=args.arch,
        feature_mode="raw",
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )
    cfg.input_shape = (args.seq_len, FEATURE_DIM)
    cfg.num_classes = n_classes
    if hasattr(cfg, "dropout"):
        cfg.dropout = args.dropout
    model = build_arch(args.arch, cfg)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5_acc")],
    )
    model.summary(print_fn=log.info)

    run_dir = ARTIFACTS / args.run_name
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "reports").mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / "checkpoints" / f"{args.arch}_best.keras"

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(ckpt_path), monitor="val_accuracy" if val_ds else "accuracy",
            save_best_only=True, mode="max", verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss" if val_ds else "loss",
            factor=0.5, patience=5, min_lr=1e-5, verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy" if val_ds else "accuracy",
            patience=12, mode="max", restore_best_weights=True, verbose=1,
        ),
    ]

    start = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=2,
    )
    elapsed = time.time() - start
    log.info("Training finished in %.1f min", elapsed / 60)

    # Save artifacts.
    (run_dir / "reports" / "config.json").write_text(json.dumps({
        "arch": args.arch,
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "epochs_run": len(history.history["loss"]),
        "n_classes": n_classes,
        "n_train": len(train_items),
        "n_val": len(val_items),
        "n_test": len(test_items),
    }, indent=2))
    (run_dir / "reports" / "history.json").write_text(json.dumps(history.history, indent=2))
    (run_dir / "word_label_map.json").write_text(json.dumps(label_map, indent=2))

    if test_ds is not None:
        log.info("Evaluating on test split…")
        results = model.evaluate(test_ds, verbose=0, return_dict=True)
        log.info("Test results: %s", results)
        (run_dir / "reports" / "test_metrics.json").write_text(json.dumps(results, indent=2))

        # Per-class accuracy.
        y_true, y_pred = [], []
        for x, y in test_ds:
            probs = model.predict(x, verbose=0)
            y_pred.extend(np.argmax(probs, axis=1).tolist())
            y_true.extend(y.numpy().tolist())
        per_class = {}
        for cls_idx, cls_name in inv.items():
            mask = [t == cls_idx for t in y_true]
            n = sum(mask)
            if n == 0:
                per_class[cls_name] = {"n": 0, "acc": None}
            else:
                correct = sum(1 for t, p, m in zip(y_true, y_pred, mask) if m and t == p)
                per_class[cls_name] = {"n": n, "acc": correct / n}
        (run_dir / "reports" / "per_class_accuracy.json").write_text(json.dumps(per_class, indent=2))
        worst = sorted(
            ((g, v) for g, v in per_class.items() if v["n"] > 0),
            key=lambda kv: kv[1]["acc"],
        )[:10]
        log.info("Worst-10 classes on test set:")
        for g, v in worst:
            log.info("  %-20s  n=%d  acc=%.2f", g, v["n"], v["acc"])

    log.info("Run dir: %s", run_dir)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--arch", default="bilstm",
                   help="Architecture: lstm, bilstm, transformer, tcn, cnn_bilstm, conformer_lite")
    p.add_argument("--run-name", default=f"word-bilstm-{int(time.time())}")
    p.add_argument("--seq-len", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.4)
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
