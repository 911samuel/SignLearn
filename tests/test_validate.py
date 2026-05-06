"""Tests for Subtask 8: dataset validation."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from backend.data.validate import (
    SEQUENCE_LEN,
    TWO_HAND_DIM,
    _parse_stem,
    _scan,
    build_class_distribution,
    check_shapes_and_integrity,
    check_subject_leakage,
    compute_feature_stats,
    validate,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _write_npy(path: Path, shape=(SEQUENCE_LEN, TWO_HAND_DIM), dtype=np.float32,
               has_nan=False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.random.rand(*shape).astype(dtype)
    if has_nan:
        arr[0, 0] = np.nan
    np.save(str(path), arr)


def _clean_dir(tmp_path: Path, n_per_split: int = 5) -> Path:
    """Populate a clean processed dir with valid vocab labels and no leakage."""
    # Use alphabet labels a-e for train (subjects 1-7), f-g for val (8-9),
    # h-i for test (10-11) — no subject appears in two splits.
    split_subjects = {"train": list(range(1, 8)), "val": [8, 9], "test": [10, 11]}
    label_map = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4}

    for split, subjects in split_subjects.items():
        for i in range(n_per_split):
            subj  = subjects[i % len(subjects)]
            label = list(label_map)[i % len(label_map)]
            stem  = f"{label}_s{subj:02d}_{i:04d}.npy"
            _write_npy(tmp_path / split / stem)

    return tmp_path


# ---------------------------------------------------------------------------
# _parse_stem
# ---------------------------------------------------------------------------

class TestParseStem:
    def test_vocab_label(self):
        result = _parse_stem("hello_s03_0042")
        assert result == ("hello", 3, 42)

    def test_digit_alias(self):
        result = _parse_stem("0_s01_0000")
        assert result == ("zero", 1, 0)

    def test_multiword_label(self):
        result = _parse_stem("good_morning_s07_0010")
        assert result == ("good_morning", 7, 10)

    def test_bad_stem_returns_none(self):
        assert _parse_stem("badname") is None
        assert _parse_stem("a_s1_1") is None   # wrong padding


# ---------------------------------------------------------------------------
# check_subject_leakage
# ---------------------------------------------------------------------------

class TestSubjectLeakage:
    def test_no_leakage(self, tmp_path):
        data = {
            "train": [(None, "a", 1), (None, "a", 2)],
            "val":   [(None, "b", 8)],
            "test":  [(None, "c", 10)],
        }
        assert check_subject_leakage(data) == []

    def test_detects_leakage(self, tmp_path):
        data = {
            "train": [(None, "a", 1)],
            "val":   [(None, "a", 1)],   # same subject in two splits
            "test":  [(None, "a", 10)],
        }
        errors = check_subject_leakage(data)
        assert len(errors) == 1
        assert "LEAKAGE" in errors[0]
        assert "s01" in errors[0]

    def test_multiple_leaks(self):
        data = {
            "train": [(None, "a", 1), (None, "a", 8)],
            "val":   [(None, "a", 1), (None, "a", 8)],
            "test":  [],
        }
        errors = check_subject_leakage(data)
        assert len(errors) == 2


# ---------------------------------------------------------------------------
# check_shapes_and_integrity
# ---------------------------------------------------------------------------

class TestShapesAndIntegrity:
    def test_valid_files_no_errors(self, tmp_path):
        npy = tmp_path / "a_s01_0000.npy"
        _write_npy(npy)
        data = {"train": [(npy, "a", 1)], "val": [], "test": []}
        errors, total = check_shapes_and_integrity(data)
        assert errors == []
        assert total == 1

    def test_wrong_shape_detected(self, tmp_path):
        npy = tmp_path / "a_s01_0000.npy"
        _write_npy(npy, shape=(30, 63))
        data = {"train": [(npy, "a", 1)], "val": [], "test": []}
        errors, _ = check_shapes_and_integrity(data)
        assert any("SHAPE" in e for e in errors)

    def test_wrong_dtype_detected(self, tmp_path):
        npy = tmp_path / "a_s01_0000.npy"
        _write_npy(npy, dtype=np.float64)
        data = {"train": [(npy, "a", 1)], "val": [], "test": []}
        errors, _ = check_shapes_and_integrity(data)
        assert any("DTYPE" in e for e in errors)

    def test_nan_detected(self, tmp_path):
        npy = tmp_path / "a_s01_0000.npy"
        _write_npy(npy, has_nan=True)
        data = {"train": [(npy, "a", 1)], "val": [], "test": []}
        errors, _ = check_shapes_and_integrity(data)
        assert any("NAN" in e for e in errors)

    def test_missing_file_detected(self, tmp_path):
        npy = tmp_path / "ghost.npy"   # does not exist
        data = {"train": [(npy, "a", 1)], "val": [], "test": []}
        errors, _ = check_shapes_and_integrity(data)
        assert any("LOAD_ERROR" in e for e in errors)


# ---------------------------------------------------------------------------
# build_class_distribution
# ---------------------------------------------------------------------------

class TestClassDistribution:
    def test_counts_correct(self):
        data = {
            "train": [(None, "a", 1), (None, "a", 2), (None, "b", 3)],
            "val":   [(None, "b", 8)],
            "test":  [],
        }
        dist = build_class_distribution(data)
        assert dist["train"]["a"] == 2
        assert dist["train"]["b"] == 1
        assert dist["val"]["b"] == 1
        assert "a" not in dist["test"]

    def test_empty_split(self):
        data = {"train": [], "val": [], "test": []}
        dist = build_class_distribution(data)
        for s in ("train", "val", "test"):
            assert dist[s] == {}


# ---------------------------------------------------------------------------
# compute_feature_stats
# ---------------------------------------------------------------------------

class TestFeatureStats:
    def test_output_keys(self, tmp_path):
        npy = tmp_path / "a_s01_0000.npy"
        _write_npy(npy)
        data = {"train": [(npy, "a", 1)], "val": [], "test": []}
        stats = compute_feature_stats(data)
        assert "mean" in stats and "std" in stats and "n_frames" in stats

    def test_mean_std_shapes(self, tmp_path):
        for i in range(3):
            _write_npy(tmp_path / f"a_s0{i+1}_000{i}.npy")
        data = {
            "train": [(tmp_path / f"a_s0{i+1}_000{i}.npy", "a", i+1) for i in range(3)],
            "val": [], "test": [],
        }
        stats = compute_feature_stats(data)
        assert len(stats["mean"]) == TWO_HAND_DIM
        assert len(stats["std"])  == TWO_HAND_DIM

    def test_n_frames(self, tmp_path):
        for i in range(4):
            _write_npy(tmp_path / f"a_s0{i+1}_000{i}.npy")
        data = {
            "train": [(tmp_path / f"a_s0{i+1}_000{i}.npy", "a", i+1) for i in range(4)],
            "val": [], "test": [],
        }
        stats = compute_feature_stats(data)
        assert stats["n_frames"] == 4 * SEQUENCE_LEN

    def test_empty_train_returns_empty(self):
        data = {"train": [], "val": [], "test": []}
        assert compute_feature_stats(data) == {}


# ---------------------------------------------------------------------------
# validate (integration)
# ---------------------------------------------------------------------------

class TestValidateIntegration:
    def test_clean_dir_passes(self, tmp_path):
        root = _clean_dir(tmp_path / "proc")
        errors = validate(processed_dir=root, artifacts_dir=tmp_path / "art")
        assert errors == [], f"Unexpected errors: {errors}"

    def test_report_written(self, tmp_path):
        root = _clean_dir(tmp_path / "proc")
        art  = tmp_path / "art"
        validate(processed_dir=root, artifacts_dir=art)
        assert (art / "validation_report.md").exists()

    def test_stats_written(self, tmp_path):
        root = _clean_dir(tmp_path / "proc")
        art  = tmp_path / "art"
        validate(processed_dir=root, artifacts_dir=art)
        stats_path = art / "feature_stats.json"
        assert stats_path.exists()
        stats = json.loads(stats_path.read_text())
        assert "mean" in stats and "std" in stats

    def test_leakage_causes_failure(self, tmp_path):
        root = tmp_path / "proc"
        # Same subject (s01) in both train and val → leakage
        _write_npy(root / "train" / "a_s01_0000.npy")
        _write_npy(root / "val"   / "b_s01_0000.npy")
        _write_npy(root / "test"  / "c_s10_0000.npy")
        errors = validate(processed_dir=root, artifacts_dir=tmp_path / "art")
        assert any("LEAKAGE" in e for e in errors)

    def test_bad_shape_causes_failure(self, tmp_path):
        root = tmp_path / "proc"
        _write_npy(root / "train" / "a_s01_0000.npy", shape=(30, 63))
        errors = validate(processed_dir=root, artifacts_dir=tmp_path / "art")
        assert any("SHAPE" in e for e in errors)

    def test_nan_causes_failure(self, tmp_path):
        root = tmp_path / "proc"
        _write_npy(root / "train" / "a_s01_0000.npy", has_nan=True)
        errors = validate(processed_dir=root, artifacts_dir=tmp_path / "art")
        assert any("NAN" in e for e in errors)


# ---------------------------------------------------------------------------
# Integration against real data (skipped if absent)
# ---------------------------------------------------------------------------

PROCESSED = Path("data/processed")
HAS_DATA  = (PROCESSED / "train").exists() and any((PROCESSED / "train").glob("*.npy"))


@pytest.mark.skipif(not HAS_DATA, reason="data/processed not populated")
class TestRealDataValidation:
    def test_real_data_passes(self, tmp_path):
        errors = validate(processed_dir=PROCESSED, artifacts_dir=tmp_path / "art")
        assert errors == [], f"Real data validation failed: {errors}"
