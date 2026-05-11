"""Validates the shape, dtype, and content of processed landmark sequences.

Covers two cases:
  1. Pre-built fixture files in tests/fixtures/processed_mini/ — always run in CI,
     exercise the full-vocab shape (30, 126) expected by the LSTM.
  2. A live webcam-extracted sample at data/processed/sample.npy — only run when
     that file exists (i.e. after running scripts/extract_landmarks.py manually).

Run all pipeline tests:
    pytest tests/test_pipeline.py -v
"""

import os
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Constants aligned with the actual two-hand pipeline shape
# ---------------------------------------------------------------------------

SEQUENCE_LENGTH = 30
LANDMARK_DIM    = 126   # both hands: 21 landmarks × 3 coords × 2 hands

_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "processed_mini"
_LIVE_SAMPLE  = Path("data/processed/sample.npy")


# ---------------------------------------------------------------------------
# 1 — Fixture-based tests (always run; no external dependencies)
# ---------------------------------------------------------------------------

def _pick_fixture() -> Path:
    """Return the path of any .npy file from the test split of processed_mini."""
    test_split = _FIXTURE_DIR / "test"
    files = sorted(test_split.glob("*.npy"))
    if not files:
        pytest.skip(
            f"No fixture files found in {test_split}. "
            "Run: python tests/generate_test_fixtures.py"
        )
    return files[0]


@pytest.fixture(scope="module")
def fixture_sample() -> np.ndarray:
    return np.load(_pick_fixture())


def test_fixture_shape(fixture_sample):
    assert fixture_sample.shape == (SEQUENCE_LENGTH, LANDMARK_DIM), (
        f"Expected ({SEQUENCE_LENGTH}, {LANDMARK_DIM}), got {fixture_sample.shape}"
    )


def test_fixture_no_nans(fixture_sample):
    assert not np.isnan(fixture_sample).any(), "NaN values found in fixture landmark data"


def test_fixture_dtype(fixture_sample):
    assert fixture_sample.dtype == np.float32


def test_fixture_has_nonzero_frames(fixture_sample):
    """At least one hand must be detected — no fixture should be all zeros."""
    nonzero_frames = int(fixture_sample.any(axis=1).sum())
    assert nonzero_frames > 0, "All frames are zero — fixture has no hand landmarks"


def test_fixture_values_in_range(fixture_sample):
    """x/y normalised coords should be in [0, 1]; z (depth) is small."""
    # Columns 63-125 = right hand (populated); check x/y pairs are in [0, 1]
    right_xy = fixture_sample[:, 63:126].reshape(SEQUENCE_LENGTH, 21, 3)[:, :, :2]
    # Ignore zero-padded left-hand slot
    assert right_xy.min() >= 0.0, "Negative x/y coordinate in right-hand slot"
    assert right_xy.max() <= 1.0, "x/y coordinate > 1.0 in right-hand slot"


def test_all_fixture_splits_have_files():
    """Each split directory must have at least one .npy file."""
    for split in ("train", "val", "test"):
        split_dir = _FIXTURE_DIR / split
        files = list(split_dir.glob("*.npy"))
        assert len(files) > 0, (
            f"Split '{split}' has no fixture files in {split_dir}. "
            "Run: python tests/generate_test_fixtures.py"
        )


# ---------------------------------------------------------------------------
# 2 — Live-sample tests (only run when a webcam-extracted sample exists)
# ---------------------------------------------------------------------------

_live_skip = pytest.mark.skipif(
    not _LIVE_SAMPLE.exists(),
    reason=f"{_LIVE_SAMPLE} not found — run scripts/extract_landmarks.py first",
)


@pytest.fixture(scope="module")
def live_sample():
    return np.load(_LIVE_SAMPLE)


@_live_skip
def test_live_sample_shape(live_sample):
    assert live_sample.shape == (SEQUENCE_LENGTH, LANDMARK_DIM), (
        f"Expected ({SEQUENCE_LENGTH}, {LANDMARK_DIM}), got {live_sample.shape}"
    )


@_live_skip
def test_live_sample_no_nans(live_sample):
    assert not np.isnan(live_sample).any()


@_live_skip
def test_live_sample_dtype(live_sample):
    assert live_sample.dtype == np.float32
