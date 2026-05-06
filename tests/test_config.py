"""Subtask 1: smoke test for TrainConfig and compact label map helpers."""

from backend.model.config import (
    FEATURE_DIM,
    SEQUENCE_LEN,
    TrainConfig,
    compact_class_names,
    compact_label_map,
    present_label_indices,
)


def test_default_input_shape():
    cfg = TrainConfig()
    assert cfg.input_shape == (SEQUENCE_LEN, FEATURE_DIM)
    assert cfg.input_shape == (30, 126)


def test_num_classes_matches_present_labels():
    cfg = TrainConfig()
    present = present_label_indices()
    assert cfg.num_classes == len(present)
    assert cfg.num_classes >= 1


def test_compact_label_map_is_zero_indexed_and_dense():
    cmap = compact_label_map()
    assert set(cmap.values()) == set(range(len(cmap)))


def test_compact_class_names_align_with_map():
    names = compact_class_names()
    assert len(names) == len(compact_label_map())
    assert all(isinstance(n, str) and n for n in names)


def test_hyperparameters_are_sane():
    cfg = TrainConfig()
    assert 0.0 <= cfg.dropout < 1.0
    assert 0.0 <= cfg.recurrent_dropout < 1.0
    assert cfg.learning_rate > 0
    assert cfg.batch_size > 0
    assert cfg.epochs > 0
    assert cfg.lstm_units == (128, 64)
