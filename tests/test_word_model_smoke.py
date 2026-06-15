"""Smoke test for the production word model.

Loads the production ONNX checkpoint at
``artifacts/checkpoints/tcn_word_best.onnx`` (tracked in git, served by
Render), runs 5 reference sequences from tests/fixtures/word_smoke/, and
asserts the correct gloss is in the top-2 predictions.

The fixtures are real test-split clips from the v3 vocabulary's
high-confidence classes (per-class acc ≥ 0.90 in word-curated-v3-64cls).
The production word model is seq_len=80; fixtures are stored at
seq_len=80 so no resampling is needed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
# Production word checkpoint (tracked in git; what Render serves).
ONNX_PATH = REPO_ROOT / "artifacts" / "checkpoints" / "tcn_word_best.onnx"
# Class-name source — same file the backend reads via
# backend.api.model_loader._WORD_VOCAB_PATH.
VOCAB_PATH = REPO_ROOT / "configs" / "word_curated_v3.txt"
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "word_smoke"

TARGET_SEQ_LEN = 80


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


def _load_vocab() -> dict[str, int]:
    """Same logic as backend.api.model_loader._load_word_class_names."""
    raw = [w.strip().lower() for w in VOCAB_PATH.read_text().splitlines()
           if w.strip() and not w.strip().startswith("#")]
    names = sorted(set(raw))
    return {n: i for i, n in enumerate(names)}


@pytest.mark.skipif(not ONNX_PATH.exists(), reason="word ONNX not present")
def test_word_model_smoke_top2():
    onnxruntime = pytest.importorskip("onnxruntime")
    label_map = _load_vocab()
    inv = {v: k for k, v in label_map.items()}

    sess = onnxruntime.InferenceSession(str(ONNX_PATH))
    inp_name = sess.get_inputs()[0].name

    fixtures = sorted(FIXTURES.glob("*.npy"))
    assert fixtures, f"no fixtures found in {FIXTURES}"

    results = []
    for f in fixtures:
        true_label = f.stem
        if true_label not in label_map:
            pytest.skip(f"fixture label {true_label!r} not in word vocab")
        arr = np.load(f)
        x = _resample(arr, TARGET_SEQ_LEN)[None, ...]
        probs = sess.run(None, {inp_name: x})[0][0]
        top2 = np.argsort(probs)[::-1][:2].tolist()
        top2_names = [inv[i] for i in top2]
        results.append((true_label, top2_names, float(probs[label_map[true_label]])))

    failures = [r for r in results if r[0] not in r[1]]
    assert not failures, f"true label not in top-2 for: {failures}"

    # Sanity: confidence on golden clips ≥ 0.05 (well above uniform 1/64).
    weak = [r for r in results if r[2] < 0.05]
    assert not weak, f"weak confidence on golden clips: {weak}"
