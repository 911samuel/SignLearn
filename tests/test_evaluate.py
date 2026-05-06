"""Subtask 5: tests for the evaluation script."""

import json
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from backend.model.architecture import build_lstm
from backend.model.config import PROCESSED_DIR, TrainConfig
from scripts.evaluate_model import evaluate


@pytest.fixture(scope="module")
def tiny_model_path(tmp_path_factory):
    """Save a randomly initialised model to a temp .h5 file."""
    tmp = tmp_path_factory.mktemp("model")
    cfg = TrainConfig(epochs=0)
    model = build_lstm(cfg)
    path = tmp / "tiny.keras"
    model.save(str(path))
    return path


def test_evaluate_produces_all_report_files(tiny_model_path, tmp_path):
    metrics = evaluate(tiny_model_path, data_dir=PROCESSED_DIR, reports_dir=tmp_path)

    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "classification_report.txt").exists()
    assert (tmp_path / "confusion_matrix.png").exists()


def test_metrics_json_has_required_keys(tiny_model_path, tmp_path):
    metrics = evaluate(tiny_model_path, data_dir=PROCESSED_DIR, reports_dir=tmp_path)

    for key in ("accuracy", "precision_macro", "recall_macro", "f1_macro",
                "test_samples", "num_classes", "class_names"):
        assert key in metrics, f"Missing key: {key}"


def test_metrics_values_are_in_valid_range(tiny_model_path, tmp_path):
    metrics = evaluate(tiny_model_path, data_dir=PROCESSED_DIR, reports_dir=tmp_path)

    for key in ("accuracy", "precision_macro", "recall_macro", "f1_macro"):
        val = metrics[key]
        assert 0.0 <= val <= 1.0, f"{key}={val} out of [0,1]"

    assert metrics["test_samples"] > 0
    assert metrics["num_classes"] == len(metrics["class_names"])
