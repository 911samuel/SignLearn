"""Subtask 3: smoke test — training completes 1 epoch on real data splits."""

import json
from pathlib import Path

import numpy as np
import pytest

from backend.model.config import ARTIFACTS_DIR, PROCESSED_DIR, TrainConfig
from scripts.train_model import train

# Use the pre-built processed fixtures so the smoke test is self-contained and
# never depends on data/processed/ being populated by a prior extraction run.
_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "processed_mini"


def test_train_one_epoch_produces_artifacts(tmp_path):
    """Run 1 epoch of real training; confirm model + history are written."""
    data_dir = _FIXTURE_DIR if _FIXTURE_DIR.exists() else PROCESSED_DIR
    config = TrainConfig(epochs=1, batch_size=16)

    history = train(config, data_dir=data_dir, out_dir=tmp_path)

    # history dict has expected keys
    assert "accuracy" in history
    assert "val_accuracy" in history
    assert "loss" in history
    assert "val_loss" in history
    assert len(history["accuracy"]) == 1

    # checkpoint produced
    assert (tmp_path / "checkpoints" / "lstm_final.keras").exists()

    # history JSON produced
    hist_path = tmp_path / "reports" / "history.json"
    assert hist_path.exists()
    with open(hist_path) as f:
        saved = json.load(f)
    assert "val_accuracy" in saved


def test_train_label_remapping_stays_in_range():
    """Compact labels produced during training must all be in [0, num_classes)."""
    import tensorflow as tf
    from backend.data.dataset import build_dataset
    from backend.model.config import compact_label_map
    from scripts.train_model import _remap_labels

    data_dir = _FIXTURE_DIR if _FIXTURE_DIR.exists() else PROCESSED_DIR
    cmap = compact_label_map(processed_dir=data_dir)
    num_classes = len(cmap)

    ds = build_dataset("train", batch_size=64, augment=False, processed_dir=data_dir)
    ds = _remap_labels(ds, cmap)

    for _, labels in ds.take(3):
        arr = labels.numpy()
        assert np.all(arr >= 0), "negative label found"
        assert np.all(arr < num_classes), f"label >= num_classes ({num_classes}) found"
