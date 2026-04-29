"""
Validates output produced by scripts/extract_landmarks.py.
Run after generating sample.npy:
    python scripts/extract_landmarks.py
    pytest tests/test_pipeline.py -v
"""

import os
import pytest
import numpy as np

SAMPLE_PATH     = "data/processed/sample.npy"
SEQUENCE_LENGTH = 30
LANDMARK_DIM    = 63

pytestmark = pytest.mark.skipif(
    not os.path.exists(SAMPLE_PATH),
    reason=f"{SAMPLE_PATH} not found — run scripts/extract_landmarks.py first",
)


@pytest.fixture(scope="module")
def sample():
    return np.load(SAMPLE_PATH)


def test_sample_shape(sample):
    assert sample.shape == (SEQUENCE_LENGTH, LANDMARK_DIM), (
        f"Expected ({SEQUENCE_LENGTH}, {LANDMARK_DIM}), got {sample.shape}"
    )


def test_no_nans(sample):
    assert not np.isnan(sample).any(), "NaN values found in landmark data"


def test_frame_count(sample):
    assert sample.shape[0] == SEQUENCE_LENGTH


def test_landmark_dim(sample):
    assert sample.shape[1] == LANDMARK_DIM


def test_dtype(sample):
    assert sample.dtype == np.float32
