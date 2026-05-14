"""Subtask 7: tf.data pipeline for train/val/test splits with optional augmentation."""

import re
from pathlib import Path

import numpy as np
import tensorflow as tf

from backend.data.augment import random_augment
from backend.data.constants import FEATURE_DIM, SEQUENCE_LEN
from backend.data.features import apply_feature_mode, output_dim
from backend.data.label_map import load_label_map, resolve_label
from backend.data.normalize import TWO_HAND_DIM, normalize_sequence

_REPO_ROOT = Path(__file__).parent.parent.parent
_PROCESSED = _REPO_ROOT / "data" / "processed"

# Re-export so callers that did `from backend.data.dataset import SEQUENCE_LEN`
# continue to work without change.
__all__ = ["SEQUENCE_LEN", "build_dataset", "list_split"]

# Regex to extract the raw label prefix from a canonical filename like
# "zero_s03_0042.npy" or "0_s03_0042.npy"
_STEM_RE = re.compile(r"^(.+)_s\d{2}_\d{4}$")


def list_split(split: str, processed_dir: Path | None = None) -> list[tuple[Path, int]]:
    """Return [(npy_path, label_idx), ...] for all files in a split directory.

    Files whose label prefix doesn't resolve to a known vocabulary entry are
    skipped with a warning so a stale or mismatched processed dir never crashes
    training.

    Args:
        split:         'train', 'val', or 'test'
        processed_dir: override the default data/processed root (for testing)

    Returns:
        Sorted list of (Path, int) pairs.
    """
    assert split in ("train", "val", "test"), f"Unknown split: {split!r}"
    root = (processed_dir or _PROCESSED) / split
    if not root.exists():
        return []

    label_map = load_label_map()
    result: list[tuple[Path, int]] = []
    skipped = 0

    for npy in sorted(root.glob("*.npy")):
        m = _STEM_RE.match(npy.stem)
        if not m:
            skipped += 1
            continue
        raw_label = m.group(1)
        vocab_label = resolve_label(raw_label)
        idx = label_map.get(vocab_label)
        if idx is None:
            skipped += 1
            continue
        result.append((npy, idx))

    if skipped:
        print(f"[dataset] {split}: skipped {skipped} files with unknown labels")

    return result


def build_dataset(
    split: str,
    batch_size: int = 32,
    augment: bool = False,
    shuffle: bool | None = None,
    processed_dir: Path | None = None,
    feature_mode: str = "raw",
    augment_probs: dict | None = None,
) -> tf.data.Dataset:
    """Build a tf.data.Dataset for a given split.

    Pipeline:
      paths/labels → load_npy+normalize → (optional) augment
      → (train) shuffle → batch(batch_size) → prefetch(AUTOTUNE)

    Args:
        split:         'train', 'val', or 'test'
        batch_size:    number of sequences per batch
        augment:       apply random_augment on-the-fly (train only)
        shuffle:       override shuffle behaviour (default: True for train only)
        processed_dir: override processed root (for testing)

    Returns:
        tf.data.Dataset yielding (sequences, labels) batches.
        sequences shape: (batch_size, 30, 126), dtype float32
        labels    shape: (batch_size,),          dtype int32
    """
    items = list_split(split, processed_dir=processed_dir)
    if not items:
        raise ValueError(f"No .npy files found for split {split!r}")

    paths  = [str(p) for p, _ in items]
    labels = [idx     for _, idx in items]

    paths_tensor  = tf.constant(paths,  dtype=tf.string)
    labels_tensor = tf.constant(labels, dtype=tf.int32)

    dataset = tf.data.Dataset.from_tensor_slices((paths_tensor, labels_tensor))

    def _load_and_normalize(path: tf.Tensor, label: tf.Tensor):
        def _py_load(p):
            # tf.numpy_function delivers string tensors as raw bytes objects
            arr = np.load(p.decode() if isinstance(p, bytes) else p.numpy().decode())
            arr = normalize_sequence(arr.astype(np.float32))
            return arr.astype(np.float32)
        seq = tf.numpy_function(_py_load, [path], tf.float32)
        seq.set_shape([SEQUENCE_LEN, TWO_HAND_DIM])
        return seq, label

    dataset = dataset.map(_load_and_normalize, num_parallel_calls=tf.data.AUTOTUNE)

    # Cache loaded+normalized sequences in memory so subsequent epochs avoid
    # disk I/O and Python overhead.  For train with augmentation the cache sits
    # *before* augmentation so each epoch still receives fresh random transforms.
    # Skip caching when augmentation is off and the split is train (not needed)
    # but always cache val/test since those datasets never change.
    _should_cache = (split != "train") or augment
    if _should_cache:
        dataset = dataset.cache()

    if augment:
        _rng = np.random.default_rng()

        def _augment(seq: tf.Tensor, label: tf.Tensor):
            def _py_aug(s):
                arr = s if isinstance(s, np.ndarray) else s.numpy()
                return random_augment(arr, rng=_rng, probs=augment_probs)
            aug = tf.numpy_function(_py_aug, [seq], tf.float32)
            aug.set_shape([SEQUENCE_LEN, TWO_HAND_DIM])
            return aug, label

        dataset = dataset.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)

    if feature_mode != "raw":
        feat_dim = output_dim(feature_mode)

        def _feature_map(seq: tf.Tensor, label: tf.Tensor):
            def _py_feat(s):
                arr = s if isinstance(s, np.ndarray) else s.numpy()
                return apply_feature_mode(arr, feature_mode)
            feat = tf.numpy_function(_py_feat, [seq], tf.float32)
            feat.set_shape([SEQUENCE_LEN, feat_dim])
            return feat, label

        dataset = dataset.map(_feature_map, num_parallel_calls=tf.data.AUTOTUNE)

    do_shuffle = (split == "train") if shuffle is None else shuffle
    if do_shuffle:
        dataset = dataset.shuffle(buffer_size=min(len(items), 2048), reshuffle_each_iteration=True)

    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset
