"""Subtask 2: tests for the LSTM model architecture."""

import numpy as np
import pytest
import tensorflow as tf

from backend.model.architecture import build_lstm
from backend.model.config import TrainConfig


@pytest.fixture(scope="module")
def cfg():
    return TrainConfig()


@pytest.fixture(scope="module")
def model(cfg):
    return build_lstm(cfg)


def test_model_builds(model):
    assert model is not None
    assert isinstance(model, tf.keras.Model)


def test_model_name(model):
    assert model.name == "signlearn_lstm"


def test_param_count_reasonable(model):
    n = model.count_params()
    assert n < 500_000, f"Model has {n:,} params — unexpectedly large"
    assert n > 10_000, f"Model has {n:,} params — suspiciously small"


def test_output_shape(model, cfg):
    batch = 4
    dummy = np.zeros((batch, *cfg.input_shape), dtype=np.float32)
    preds = model.predict(dummy, verbose=0)
    assert preds.shape == (batch, cfg.num_classes)


def test_output_sums_to_one(model, cfg):
    dummy = np.zeros((2, *cfg.input_shape), dtype=np.float32)
    preds = model.predict(dummy, verbose=0)
    np.testing.assert_allclose(preds.sum(axis=1), 1.0, atol=1e-5)


def test_masking_layer_present(model):
    layer_names = [l.name for l in model.layers]
    assert "masking" in layer_names


def test_loss_is_sparse_categorical(model):
    assert "sparse_categorical_crossentropy" in model.loss
