"""Export a trained Keras checkpoint to ONNX with a numerical-parity check.

Usage
-----
python backend/scripts/export_onnx.py \
    --in  artifacts/runs/bilstm-v2-36cls/checkpoints/bilstm_best.keras \
    --out artifacts/runs/bilstm-v2-36cls/bilstm_best.onnx

Verification samples are drawn from the val split (via :func:`list_split`) so
the parity check exercises the same normalize/feature pipeline used in
production. Aborts with non-zero exit code if max |Δ| ≥ ``--atol``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from backend.data.constants import FEATURE_DIM, SEQUENCE_LEN  # noqa: E402
from backend.data.dataset import list_split  # noqa: E402
from backend.data.features import apply_feature_mode, output_dim  # noqa: E402
from backend.data.normalize import normalize_sequence  # noqa: E402


def _resolve_feature_mode(keras_path: Path) -> str:
    """Read feature_mode from the run's config.json, falling back to 'raw'."""
    cfg_path = keras_path.parent.parent / "reports" / "config.json"
    if not cfg_path.exists():
        return "raw"
    import json
    cfg = json.loads(cfg_path.read_text())
    return cfg.get("feature_mode", "raw")


def _build_validation_batch(n: int, feature_mode: str, seq_len: int = SEQUENCE_LEN) -> np.ndarray:
    """Sample ``n`` real validation sequences and apply the feature pipeline."""
    items = list_split("val") if seq_len == SEQUENCE_LEN else []
    if not items:
        # Fall back to synthetic noise — better than crashing the export.
        # The word model has no letter val split; synthetic is fine for parity.
        D = output_dim(feature_mode)
        return np.random.randn(n, seq_len, D).astype(np.float32)
    idx = np.random.default_rng(seed=42).choice(len(items), size=min(n, len(items)), replace=False)
    batch = []
    for j in idx:
        path, _ = items[int(j)]
        raw = np.load(path).astype(np.float32)
        if raw.shape != (SEQUENCE_LEN, FEATURE_DIM):
            continue
        norm = normalize_sequence(raw)
        batch.append(apply_feature_mode(norm, feature_mode))
    return np.stack(batch, axis=0).astype(np.float32)


def export(in_path: Path, out_path: Path, opset: int, atol: float, n_samples: int) -> dict:
    import os
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import tensorflow as tf
    import tf2onnx  # type: ignore[import-not-found]

    feature_mode = _resolve_feature_mode(in_path)
    print(f"Detected feature_mode={feature_mode}; feature_dim={output_dim(feature_mode)}")

    model = tf.keras.models.load_model(str(in_path))
    print(f"Loaded {in_path.name}: input_shape={model.input_shape}, params={model.count_params():,}")

    # Use the model's actual sequence length so the word model (T=80) exports
    # correctly alongside the letter model (T=30).
    model_seq_len = model.input_shape[1] or SEQUENCE_LEN

    # Dynamic batch axis so onnxruntime can serve batch>1 if we ever want to.
    spec = (
        tf.TensorSpec(
            (None, model_seq_len, output_dim(feature_mode)),
            tf.float32,
            name=model.inputs[0].name.split(":")[0],
        ),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    onnx_model, _ = tf2onnx.convert.from_keras(
        model,
        input_signature=spec,
        opset=opset,
        output_path=str(out_path),
    )
    print(f"ONNX written → {out_path}")

    # Parity check.
    from backend.api.onnx_runner import OnnxRunner
    runner = OnnxRunner(out_path)
    batch = _build_validation_batch(n_samples, feature_mode, seq_len=model_seq_len)
    print(f"Parity batch: shape={batch.shape}")

    keras_out = model.predict(batch, verbose=0)
    onnx_out  = runner.predict(batch)
    diff = float(np.max(np.abs(keras_out - onnx_out)))
    rel = float(np.max(np.abs(keras_out - onnx_out) / (np.abs(keras_out) + 1e-8)))

    result = {
        "input": str(in_path),
        "output": str(out_path),
        "feature_mode": feature_mode,
        "n_validation_samples": int(batch.shape[0]),
        "max_abs_diff": diff,
        "max_rel_diff": rel,
        "atol": atol,
        "passed": diff < atol,
    }
    print()
    print(f"Max |Δ|     : {diff:.3e}")
    print(f"Max rel Δ   : {rel:.3e}")
    print(f"Tolerance   : {atol:.3e}")
    if not result["passed"]:
        print(f"❌ Parity FAILED — onnx output diverges from Keras by {diff:.3e}")
        sys.exit(1)
    print("✅ Parity passed.")
    return result


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Export a Keras checkpoint to ONNX")
    p.add_argument("--in",  dest="in_path",  type=Path, required=True,
                   help="Path to <arch>_best.keras")
    p.add_argument("--out", dest="out_path", type=Path, default=None,
                   help="Output .onnx path (default: same dir as input)")
    p.add_argument("--opset", type=int, default=17, help="ONNX opset (default: 17)")
    p.add_argument("--atol",  type=float, default=1e-4, help="Max allowed |Δ|")
    p.add_argument("--n",     type=int, default=100, help="Validation batch size for parity check")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    out_path = args.out_path or args.in_path.with_suffix(".onnx")
    export(args.in_path, out_path, opset=args.opset, atol=args.atol, n_samples=args.n)
