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
from backend.data.augment import AUG_PROFILES, random_augment
from backend.data.features import apply_feature_mode, output_dim as feature_dim
from backend.data.mixup import same_class_mixup
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
    # Normalise glosses to lowercase — filenames are lowercased by the
    # ASL Citizen extractor, while curated word lists may use uppercase.
    glosses = sorted({
        w.strip().lower() for w in curated_path.read_text().splitlines()
        if w.strip() and not w.startswith("#")
    })
    return {g: i for i, g in enumerate(glosses)}


def _gloss_from_filename(path: Path) -> str:
    """Recover the gloss from a SignLearn .npy filename.

    Naming convention: ``<gloss>_<signer_token>_<idx>.npy`` where the signer
    token starts with 's' (e.g. ``s05`` for WLASL, ``syt0a3f`` for YouTube,
    ``sacABC123`` for ASL Citizen) and idx is a 4-digit zero-padded integer.
    The gloss itself may contain underscores (e.g. ``thank_you``).
    """
    stem = path.stem.lower()
    parts = stem.split("_")
    # Strict pattern: last token is 4-digit idx; second-to-last starts with 's'
    if (len(parts) >= 3 and parts[-1].isdigit()
            and parts[-2].startswith("s") and len(parts[-2]) > 1):
        return "_".join(parts[:-2])
    # Fallback: drop the last 2 tokens
    return "_".join(parts[:-2]) if len(parts) >= 3 else stem


STORED_SEQ_LEN = 80  # native length of data/processed/words/ samples


def list_split(split: str, label_map: dict[str, int], seq_len: int) -> list[tuple[Path, int]]:
    """List samples. Stored arrays are (STORED_SEQ_LEN, FEATURE_DIM); the
    loader resamples to ``seq_len`` at training time (see _resample)."""
    split_dir = DATA_ROOT / split
    items: list[tuple[Path, int]] = []
    skipped_unknown = 0
    skipped_shape = 0
    for npy in split_dir.glob("*.npy"):
        gloss = _gloss_from_filename(npy).lower()
        if gloss not in label_map:
            skipped_unknown += 1
            continue
        arr = np.load(npy, mmap_mode="r")
        if arr.shape != (STORED_SEQ_LEN, FEATURE_DIM):
            skipped_shape += 1
            continue
        items.append((npy, label_map[gloss]))
    if skipped_unknown:
        log.warning("%s: skipped %d file(s) with gloss not in label_map", split, skipped_unknown)
    if skipped_shape:
        log.warning("%s: skipped %d file(s) with wrong shape (expected (%d, %d))",
                    split, skipped_shape, STORED_SEQ_LEN, FEATURE_DIM)
    return items


def _resample_to(arr: np.ndarray, target_T: int) -> np.ndarray:
    """Resample a stored (STORED_SEQ_LEN, D) sequence to (target_T, D).

    Trims trailing all-zero (padding) frames first, then linearly interpolates
    the *nonzero core* to ``target_T``. If the entire sequence is zero
    (degenerate), returns zeros at the target length.
    """
    if arr.shape[0] == target_T:
        return arr.astype(np.float32)
    # find last frame with any nonzero coord
    nz = np.any(arr != 0, axis=1)
    if not nz.any():
        return np.zeros((target_T, arr.shape[1]), dtype=np.float32)
    last = int(np.where(nz)[0].max()) + 1  # number of active frames
    core = arr[:last]
    if last == 1:
        return np.broadcast_to(core, (target_T, arr.shape[1])).astype(np.float32).copy()
    src_t = np.linspace(0.0, 1.0, last)
    dst_t = np.linspace(0.0, 1.0, target_T)
    out = np.empty((target_T, arr.shape[1]), dtype=np.float32)
    for d in range(arr.shape[1]):
        out[:, d] = np.interp(dst_t, src_t, core[:, d])
    return out


def build_dataset(
    items: list[tuple[Path, int]],
    seq_len: int,
    batch_size: int,
    shuffle: bool,
    aug_profile: str | None = None,
    feature_mode: str = "raw",
) -> tf.data.Dataset:
    """Build a tf.data pipeline.

    aug_profile (None for val/test):
      - None / unset       — no augmentation
      - "baseline"         — TRAINING_PROBS
      - "timewarp"         — TRAINING_PROBS + time_warp p=0.5
      - "mixup_sameclass"  — TRAINING_PROBS + batch-level same-class mixup
      - "timewarp+mixup"   — both
    """
    paths = [str(p) for p, _ in items]
    labels = [int(y) for _, y in items]

    probs = AUG_PROFILES.get(aug_profile) if aug_profile else None
    use_mixup = aug_profile in {"mixup_sameclass", "timewarp+mixup"}
    out_dim = feature_dim(feature_mode)

    def _load(path: tf.Tensor, label: tf.Tensor):
        def _np_load(p):
            arr = np.load(p.decode()).astype(np.float32)
            if arr.shape[0] != seq_len:
                arr = _resample_to(arr, seq_len)
            return arr
        seq = tf.numpy_function(_np_load, [path], tf.float32)
        seq.set_shape([seq_len, FEATURE_DIM])
        return seq, label

    def _augment_one(seq: tf.Tensor, label: tf.Tensor):
        def _aug(x):
            return random_augment(x.astype(np.float32), probs=probs).astype(np.float32)
        out = tf.numpy_function(_aug, [seq], tf.float32)
        out.set_shape([seq_len, FEATURE_DIM])
        return out, label

    def _featurize(seq: tf.Tensor, label: tf.Tensor):
        def _fm(x):
            return apply_feature_mode(x.astype(np.float32), feature_mode).astype(np.float32)
        out = tf.numpy_function(_fm, [seq], tf.float32)
        out.set_shape([seq_len, out_dim])
        return out, label

    def _mixup_batch(seqs: tf.Tensor, labels_b: tf.Tensor):
        def _mx(x, y):
            mx, my = same_class_mixup(x, y, alpha=0.2)
            return mx.astype(np.float32), my.astype(np.int64)
        mx, my = tf.numpy_function(_mx, [seqs, labels_b], [tf.float32, tf.int64])
        mx.set_shape([None, seq_len, FEATURE_DIM])
        my.set_shape([None])
        return mx, my

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(items), 2048), reshuffle_each_iteration=True)
    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    if probs is not None:
        ds = ds.map(_augment_one, num_parallel_calls=tf.data.AUTOTUNE)
    # Mixup is applied to RAW (T, 126) before featurization so velocity/angles
    # are computed on the blended sequence.
    ds = ds.batch(batch_size)
    if use_mixup:
        ds = ds.map(_mixup_batch, num_parallel_calls=tf.data.AUTOTUNE)
    if feature_mode != "raw":
        ds = ds.unbatch().map(_featurize, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args) -> None:
    words_path = Path(args.words_file) if args.words_file else CURATED_PATH
    print(f"Using vocabulary: {words_path}")
    label_map = load_label_map(words_path)
    n_classes = len(label_map)
    log.info("Classes: %d", n_classes)

    train_items = list_split("train", label_map, args.seq_len)
    val_items = list_split("val", label_map, args.seq_len)
    test_items = list_split("test", label_map, args.seq_len)
    log.info("Items — train=%d  val=%d  test=%d", len(train_items), len(val_items), len(test_items))

    if args.subsample_class and args.subsample_pct is not None:
        # Subsample only the specified class in TRAIN; val/test untouched.
        target_idx = label_map.get(args.subsample_class.lower())
        if target_idx is None:
            sys.exit(f"--subsample-class {args.subsample_class!r}: unknown class")
        target = [it for it in train_items if it[1] == target_idx]
        other  = [it for it in train_items if it[1] != target_idx]
        rng = np.random.default_rng(args.seed)
        keep_n = max(1, int(round(len(target) * args.subsample_pct)))
        sel = rng.choice(len(target), size=keep_n, replace=False)
        target = [target[i] for i in sel]
        train_items = other + target
        log.info("Subsampled '%s' to %d/%d (pct=%.2f). New train total: %d",
                 args.subsample_class, keep_n, len(target) + (keep_n - keep_n),
                 args.subsample_pct, len(train_items))

    if not train_items:
        sys.exit("No training data found. Run extract_wlasl_landmarks.py first.")

    # Class-balance check.
    from collections import Counter
    counts = Counter(y for _, y in train_items)
    inv = {v: k for k, v in label_map.items()}
    weak = [(inv[c], n) for c, n in counts.items() if n < 5]
    if weak:
        log.warning("Classes with <5 train samples: %s", weak)

    train_ds = build_dataset(
        train_items, args.seq_len, args.batch_size, shuffle=True,
        aug_profile=args.aug_profile if args.aug_profile != "none" else None,
        feature_mode=args.feature_mode,
    )
    val_ds = build_dataset(
        val_items, args.seq_len, args.batch_size, shuffle=False,
        feature_mode=args.feature_mode,
    ) if val_items else None
    test_ds = build_dataset(
        test_items, args.seq_len, args.batch_size, shuffle=False,
        feature_mode=args.feature_mode,
    ) if test_items else None

    cfg = TrainConfig(
        arch_name=args.arch,
        feature_mode=args.feature_mode,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )
    cfg.input_shape = (args.seq_len, feature_dim(args.feature_mode))
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
        "aug_profile": args.aug_profile,
        "feature_mode": args.feature_mode,
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
    p.add_argument("--words-file", default=None,
                   help="Path to a newline-delimited gloss list. "
                        f"Default: {CURATED_PATH.relative_to(REPO_ROOT)}")
    p.add_argument("--run-name", default=f"word-bilstm-{int(time.time())}")
    p.add_argument("--seq-len", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.4)
    p.add_argument("--aug-profile", default="none",
                   choices=["none", "baseline", "timewarp",
                            "mixup_sameclass", "timewarp+mixup"],
                   help="Augmentation profile (default: none — matches the "
                        "existing word-aslc-tcn-78cls-v1 baseline).")
    p.add_argument("--feature-mode", default="raw",
                   choices=["raw", "raw+velocity", "raw+velocity+angles"],
                   help="Reserved for step 4. Currently only 'raw' is wired "
                        "through the data pipeline.")
    p.add_argument("--subsample-class", default=None,
                   help="If set, subsample this class's training data only.")
    p.add_argument("--subsample-pct", type=float, default=None,
                   help="Fraction of --subsample-class samples to keep (0,1].")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
