"""Tests for Subtask 7: tf.data loader."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from backend.data.normalize import TWO_HAND_DIM
from backend.data.dataset import SEQUENCE_LEN, build_dataset, list_split
from backend.data.label_map import load_label_map

PROCESSED = Path("data/processed")
HAS_DATA  = (PROCESSED / "train").exists() and any((PROCESSED / "train").glob("*.npy"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_processed_dir(tmp_path: Path, split: str, n: int = 10) -> Path:
    """Write n synthetic .npy files into tmp_path/<split>/ for testing."""
    from backend.data.label_map import _LABEL_MAP_PATH
    import json, shutil

    root = tmp_path / split
    root.mkdir(parents=True)

    label_map = load_label_map()
    labels = list(label_map.keys())[:n]

    for i, label in enumerate(labels):
        stem  = f"{label}_s01_{i:04d}.npy"
        arr   = np.random.rand(SEQUENCE_LEN, TWO_HAND_DIM).astype(np.float32)
        np.save(str(root / stem), arr)

    return tmp_path


# ---------------------------------------------------------------------------
# list_split
# ---------------------------------------------------------------------------

class TestListSplit:
    def test_returns_list(self, tmp_path):
        _make_processed_dir(tmp_path, "train", n=5)
        items = list_split("train", processed_dir=tmp_path)
        assert isinstance(items, list)
        assert len(items) == 5

    def test_items_are_path_int_tuples(self, tmp_path):
        _make_processed_dir(tmp_path, "train", n=3)
        items = list_split("train", processed_dir=tmp_path)
        for path, idx in items:
            assert isinstance(path, Path)
            assert isinstance(idx, int)

    def test_label_indices_in_range(self, tmp_path):
        _make_processed_dir(tmp_path, "train", n=5)
        label_map = load_label_map()
        items = list_split("train", processed_dir=tmp_path)
        for _, idx in items:
            assert 0 <= idx < len(label_map)

    def test_empty_dir_returns_empty(self, tmp_path):
        (tmp_path / "val").mkdir()
        assert list_split("val", processed_dir=tmp_path) == []

    def test_invalid_split_raises(self, tmp_path):
        with pytest.raises(AssertionError):
            list_split("bogus", processed_dir=tmp_path)

    def test_digit_alias_resolved(self, tmp_path):
        """Files named '0_s01_0000.npy' should resolve to label index for 'zero'."""
        split_dir = tmp_path / "train"
        split_dir.mkdir()
        arr = np.zeros((SEQUENCE_LEN, TWO_HAND_DIM), dtype=np.float32)
        np.save(str(split_dir / "0_s01_0000.npy"), arr)

        items = list_split("train", processed_dir=tmp_path)
        assert len(items) == 1
        label_map = load_label_map()
        assert items[0][1] == label_map["zero"]


# ---------------------------------------------------------------------------
# build_dataset — shapes and dtypes
# ---------------------------------------------------------------------------

class TestBuildDataset:
    def test_batch_shape(self, tmp_path):
        _make_processed_dir(tmp_path, "train", n=10)
        ds = build_dataset("train", batch_size=4, processed_dir=tmp_path)
        seq, lab = next(iter(ds))
        assert seq.shape == (4, SEQUENCE_LEN, TWO_HAND_DIM), f"Got {seq.shape}"

    def test_sequence_dtype(self, tmp_path):
        _make_processed_dir(tmp_path, "train", n=5)
        ds = build_dataset("train", batch_size=4, processed_dir=tmp_path)
        seq, _ = next(iter(ds))
        assert seq.dtype == tf.float32

    def test_label_dtype(self, tmp_path):
        _make_processed_dir(tmp_path, "train", n=5)
        ds = build_dataset("train", batch_size=4, processed_dir=tmp_path)
        _, lab = next(iter(ds))
        assert lab.dtype == tf.int32

    def test_label_values_in_range(self, tmp_path):
        _make_processed_dir(tmp_path, "train", n=10)
        label_map = load_label_map()
        ds = build_dataset("train", batch_size=10, processed_dir=tmp_path)
        _, lab = next(iter(ds))
        assert tf.reduce_all(lab >= 0)
        assert tf.reduce_all(lab < len(label_map))

    def test_no_nan_in_sequences(self, tmp_path):
        _make_processed_dir(tmp_path, "train", n=5)
        ds = build_dataset("train", batch_size=5, processed_dir=tmp_path)
        seq, _ = next(iter(ds))
        assert not tf.reduce_any(tf.math.is_nan(seq))

    def test_augment_changes_values(self, tmp_path):
        """With augment=True the batches should not be identical across epochs."""
        _make_processed_dir(tmp_path, "train", n=8)
        ds = build_dataset("train", batch_size=8, augment=True,
                           shuffle=False, processed_dir=tmp_path)
        it = iter(ds)
        b1, _ = next(it)
        # Re-iterate to get a second epoch
        it2 = iter(ds)
        b2, _ = next(it2)
        # Augmentation is random — batches very likely differ
        assert not tf.reduce_all(tf.equal(b1, b2))

    def test_val_no_augment(self, tmp_path):
        """Val dataset with same data twice should produce identical batches."""
        _make_processed_dir(tmp_path, "val", n=6)
        ds1 = build_dataset("val", batch_size=6, augment=False,
                             shuffle=False, processed_dir=tmp_path)
        ds2 = build_dataset("val", batch_size=6, augment=False,
                             shuffle=False, processed_dir=tmp_path)
        seq1, _ = next(iter(ds1))
        seq2, _ = next(iter(ds2))
        np.testing.assert_allclose(seq1.numpy(), seq2.numpy(), atol=1e-5)

    def test_empty_split_raises(self, tmp_path):
        (tmp_path / "test").mkdir()
        with pytest.raises(ValueError, match="No .npy files"):
            build_dataset("test", processed_dir=tmp_path)

    def test_partial_last_batch(self, tmp_path):
        """When N is not divisible by batch_size the last batch is smaller."""
        _make_processed_dir(tmp_path, "train", n=7)
        ds = build_dataset("train", batch_size=4, shuffle=False, processed_dir=tmp_path)
        batches = list(ds)
        sizes = [b[0].shape[0] for b in batches]
        assert sum(sizes) == 7
        assert sizes[-1] == 3  # 7 % 4

    def test_prefetch_element_spec(self, tmp_path):
        _make_processed_dir(tmp_path, "train", n=5)
        ds = build_dataset("train", batch_size=5, processed_dir=tmp_path)
        seq_spec, lab_spec = ds.element_spec
        assert seq_spec.shape == (None, SEQUENCE_LEN, TWO_HAND_DIM)
        assert lab_spec.dtype == tf.int32


# ---------------------------------------------------------------------------
# Integration: real processed data (skipped when absent)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_DATA, reason="data/processed/train not populated")
class TestRealData:
    def test_train_batch_shape(self):
        ds = build_dataset("train", batch_size=32)
        seq, lab = next(iter(ds))
        assert seq.shape == (32, SEQUENCE_LEN, TWO_HAND_DIM)
        assert lab.shape == (32,)

    def test_throughput(self):
        """Sanity check: pipeline stays within an order of magnitude of expected.

        Threshold is intentionally loose — load + per-sample normalization is on
        the hot path and absolute speed varies a lot under shared CPU load.
        """
        import time
        ds = build_dataset("train", batch_size=32)
        start = time.perf_counter()
        count = 0
        for _ in ds:
            count += 1
            if count >= 20:
                break
        elapsed = time.perf_counter() - start
        bps = count / elapsed
        print(f"\nThroughput: {bps:.1f} batches/sec")
        assert bps >= 3, f"Throughput too low: {bps:.1f} batches/sec"

    def test_val_labels_in_range(self):
        label_map = load_label_map()
        ds = build_dataset("val", batch_size=64)
        for _, lab in ds:
            assert tf.reduce_all(lab >= 0)
            assert tf.reduce_all(lab < len(label_map))
