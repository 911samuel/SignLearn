"""Subtask 7: inference loading and output shape tests."""

import numpy as np
import pytest
import tensorflow as tf

from backend.model.config import CHECKPOINTS_DIR, FEATURE_DIM, SEQUENCE_LEN, TrainConfig


@pytest.fixture(scope="module")
def trained_model():
    path = CHECKPOINTS_DIR / "lstm_final.keras"
    if not path.exists():
        pytest.skip(f"Trained model not found at {path} — run train_model.py first")
    return tf.keras.models.load_model(str(path))


def test_model_loads(trained_model):
    assert trained_model is not None


def test_output_shape_single_sample(trained_model):
    x = np.zeros((1, SEQUENCE_LEN, FEATURE_DIM), dtype=np.float32)
    out = trained_model.predict(x, verbose=0)
    # Shape must be (1, N) where N is however many classes this model was trained on.
    assert len(out.shape) == 2
    assert out.shape[0] == 1
    assert out.shape[1] >= 1


def test_output_sums_to_one(trained_model):
    x = np.zeros((4, SEQUENCE_LEN, FEATURE_DIM), dtype=np.float32)
    out = trained_model.predict(x, verbose=0)
    np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-5)


def test_output_values_are_probabilities(trained_model):
    x = np.random.rand(8, SEQUENCE_LEN, FEATURE_DIM).astype(np.float32)
    out = trained_model.predict(x, verbose=0)
    assert np.all(out >= 0), "negative probability"
    assert np.all(out <= 1), "probability > 1"


def test_profile_stats_structure():
    from scripts.profile_inference import profile
    path = CHECKPOINTS_DIR / "lstm_final.keras"
    if not path.exists():
        pytest.skip("Trained model not found")
    stats = profile(path, n_runs=20)
    for key in ("mean_ms", "p50_ms", "p95_ms", "p99_ms", "throughput_fps"):
        assert key in stats
        assert stats[key] > 0
