"""Subtask 2: model loader parity tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from backend.api.inference import predict
from backend.api.model_loader import get_class_names, is_loaded, load_model
from backend.model.config import compact_class_names

FIXTURE_DIR = Path("tests/fixtures/processed_mini/train")


@pytest.fixture(scope="module", autouse=True)
def loaded_model():
    load_model()


def _first_fixture() -> np.ndarray:
    npy_files = sorted(FIXTURE_DIR.glob("*.npy"))
    assert npy_files, f"No .npy files found in {FIXTURE_DIR}"
    return np.load(npy_files[0])


def test_model_loaded():
    assert is_loaded()


def _skip_if_no_class_names():
    """Skip tests that need a reconciled model + label mapping when names are absent."""
    if not get_class_names():
        pytest.skip(
            "compact_class_names() is empty — no processed data in data/processed/. "
            "Run generate_test_fixtures.py and train_model.py first."
        )


def test_class_names_match_phase2():
    """Label decoder must match the Phase 2 compact_class_names exactly."""
    _skip_if_no_class_names()
    assert get_class_names() == compact_class_names()


def test_predict_returns_valid_label():
    _skip_if_no_class_names()
    seq = _first_fixture()
    label, conf = predict(seq)
    assert label in get_class_names(), f"Unexpected label: {label!r}"
    assert 0.0 <= conf <= 1.0, f"Confidence out of range: {conf}"


def test_predict_wrong_shape_raises():
    bad = np.zeros((15, 126), dtype=np.float32)
    with pytest.raises(ValueError, match="Expected shape"):
        predict(bad)


def test_predict_confidence_is_float():
    _skip_if_no_class_names()
    seq = _first_fixture()
    _, conf = predict(seq)
    assert isinstance(conf, float)
