"""Smoke test for the curated v3 word model.

Loads the ONNX checkpoint at
artifacts/runs/word-curated-v3-64cls/tcn_word_v3.onnx, runs 5 reference
sequences from tests/fixtures/word_smoke/, and asserts the correct gloss
is in the top-2 predictions.

The fixtures are real test-split clips from the v3 vocabulary's
high-confidence classes (per-class acc ≥ 0.90 in word-curated-v3-64cls).
The model is stored at seq_len=120; fixtures are stored at seq_len=80 and
get linearly resampled to 120 before inference (matches the training
pipeline in train_word_model.py:_resample_to).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = REPO_ROOT / "artifacts" / "runs" / "word-curated-v3-64cls-seed1"
ONNX_PATH = RUN_DIR / "tcn_word_v3_seed1.onnx"
LABEL_MAP = RUN_DIR / "word_label_map.json"
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "word_smoke"

TARGET_SEQ_LEN = 120


def _resample(arr: np.ndarray, target_T: int) -> np.ndarray:
    """Match the training-pipeline resampler (train_word_model._resample_to)."""
    if arr.shape[0] == target_T:
        return arr.astype(np.float32)
    nz = np.any(arr != 0, axis=1)
    if not nz.any():
        return np.zeros((target_T, arr.shape[1]), dtype=np.float32)
    last = int(np.where(nz)[0].max()) + 1
    core = arr[:last]
    src_t = np.linspace(0.0, 1.0, last)
    dst_t = np.linspace(0.0, 1.0, target_T)
    out = np.empty((target_T, arr.shape[1]), dtype=np.float32)
    for d in range(arr.shape[1]):
        out[:, d] = np.interp(dst_t, src_t, core[:, d])
    return out


@pytest.mark.skipif(not ONNX_PATH.exists(), reason="v3 ONNX not built — run export_onnx.py first")
def test_word_model_smoke_top2():
    onnxruntime = pytest.importorskip("onnxruntime")
    label_map = json.loads(LABEL_MAP.read_text())
    inv = {v: k for k, v in label_map.items()}

    sess = onnxruntime.InferenceSession(str(ONNX_PATH))
    inp_name = sess.get_inputs()[0].name

    fixtures = sorted(FIXTURES.glob("*.npy"))
    assert fixtures, f"no fixtures found in {FIXTURES}"

    results = []
    for f in fixtures:
        true_label = f.stem
        if true_label not in label_map:
            pytest.skip(f"fixture label {true_label!r} not in v3 vocab")
        arr = np.load(f)
        x = _resample(arr, TARGET_SEQ_LEN)[None, ...]
        probs = sess.run(None, {inp_name: x})[0][0]
        top2 = np.argsort(probs)[::-1][:2].tolist()
        top2_names = [inv[i] for i in top2]
        results.append((true_label, top2_names, float(probs[label_map[true_label]])))

    failures = [r for r in results if r[0] not in r[1]]
    assert not failures, f"true label not in top-2 for: {failures}"

    # Sanity: each true label must score at least 0.05 probability (well above
    # uniform 1/64 = 0.016). Catches regressions where the model loads but
    # confidence on known-good clips collapses.
    weak = [r for r in results if r[2] < 0.05]
    assert not weak, f"weak confidence on golden clips: {weak}"
