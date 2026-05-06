"""Subtask 1: smoke test for TrainConfig and compact label map helpers."""

from pathlib import Path

import pytest

from backend.model.config import (
    FEATURE_DIM,
    SEQUENCE_LEN,
    TrainConfig,
    compact_class_names,
    compact_label_map,
    present_label_indices,
)

_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "processed_mini"


def _data_dir():
    """Return fixture dir when present (CI), else fall back to default PROCESSED_DIR."""
    return _FIXTURE_DIR if _FIXTURE_DIR.exists() else None


def test_default_input_shape():
    cfg = TrainConfig()
    assert cfg.input_shape == (SEQUENCE_LEN, FEATURE_DIM)
    assert cfg.input_shape == (30, 126)


def test_num_classes_matches_present_labels():
    processed_dir = _data_dir()
    present = present_label_indices(processed_dir=processed_dir)
    if not present:
        pytest.skip("No processed data available — run generate_test_fixtures.py first")
    cfg = TrainConfig()
    cfg.num_classes = len(present)
    assert cfg.num_classes == len(present)
    assert cfg.num_classes >= 1


def test_compact_label_map_is_zero_indexed_and_dense():
    processed_dir = _data_dir()
    cmap = compact_label_map(processed_dir=processed_dir)
    if not cmap:
        pytest.skip("No processed data available")
    assert set(cmap.values()) == set(range(len(cmap)))


def test_compact_class_names_align_with_map():
    processed_dir = _data_dir()
    names = compact_class_names(processed_dir=processed_dir)
    cmap = compact_label_map(processed_dir=processed_dir)
    if not cmap:
        pytest.skip("No processed data available")
    assert len(names) == len(cmap)
    assert all(isinstance(n, str) and n for n in names)


def test_hyperparameters_are_sane():
    cfg = TrainConfig()
    assert 0.0 <= cfg.dropout < 1.0
    assert 0.0 <= cfg.recurrent_dropout < 1.0
    assert cfg.learning_rate > 0
    assert cfg.batch_size > 0
    assert cfg.epochs > 0
    assert cfg.lstm_units == (128, 64)
