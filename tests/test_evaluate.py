"""Subtask 5: tests for the evaluation script."""

import json
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from backend.model.architecture import build_lstm
from backend.model.config import PROCESSED_DIR, TrainConfig, compact_label_map
from scripts.evaluate_model import evaluate

_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "processed_mini"


def _data_dir() -> Path:
    return _FIXTURE_DIR if _FIXTURE_DIR.exists() else PROCESSED_DIR


@pytest.fixture(scope="module")
def tiny_model_path(tmp_path_factory):
    """Save a randomly initialised model with the right num_classes for the data."""
    tmp = tmp_path_factory.mktemp("model")
    data_dir = _data_dir()
    cmap = compact_label_map(processed_dir=data_dir)
    num_classes = len(cmap) if cmap else 1
    cfg = TrainConfig(epochs=0, num_classes=num_classes)
    model = build_lstm(cfg)
    path = tmp / "tiny.keras"
    model.save(str(path))
    return path


def test_evaluate_produces_all_report_files(tiny_model_path, tmp_path):
    data_dir = _data_dir()
    if not list((data_dir / "test").glob("*.npy") if (data_dir / "test").exists() else []):
        pytest.skip("No test split data — run generate_test_fixtures.py first")
    metrics = evaluate(tiny_model_path, data_dir=data_dir, reports_dir=tmp_path)

    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "classification_report.txt").exists()
    assert (tmp_path / "confusion_matrix.png").exists()


def test_metrics_json_has_required_keys(tiny_model_path, tmp_path):
    data_dir = _data_dir()
    if not list((data_dir / "test").glob("*.npy") if (data_dir / "test").exists() else []):
        pytest.skip("No test split data — run generate_test_fixtures.py first")
    metrics = evaluate(tiny_model_path, data_dir=data_dir, reports_dir=tmp_path)

    for key in ("accuracy", "precision_macro", "recall_macro", "f1_macro",
                "test_samples", "num_classes", "class_names"):
        assert key in metrics, f"Missing key: {key}"


def test_metrics_values_are_in_valid_range(tiny_model_path, tmp_path):
    data_dir = _data_dir()
    if not list((data_dir / "test").glob("*.npy") if (data_dir / "test").exists() else []):
        pytest.skip("No test split data — run generate_test_fixtures.py first")
    metrics = evaluate(tiny_model_path, data_dir=data_dir, reports_dir=tmp_path)

    for key in ("accuracy", "precision_macro", "recall_macro", "f1_macro"):
        val = metrics[key]
        assert 0.0 <= val <= 1.0, f"{key}={val} out of [0,1]"

    assert metrics["test_samples"] > 0
    assert metrics["num_classes"] == len(metrics["class_names"])
