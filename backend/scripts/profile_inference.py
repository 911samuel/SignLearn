"""Phase 2 — Subtask 7: Inference latency profiling and final model export.

Runs 1000 single-sample forward passes (CPU, no batching) and reports
mean / p50 / p95 / p99 latency + throughput. Target: p95 < 500 ms.

Writes:
  artifacts/reports/inference_profile.md
  artifacts/checkpoints/lstm_final.keras   (canonical Phase 3 handoff)

Usage
-----
python backend/scripts/profile_inference.py
python backend/scripts/profile_inference.py --model artifacts/checkpoints/lstm_best.keras --n 500
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

_REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from backend.model.config import CHECKPOINTS_DIR, FEATURE_DIM, REPORTS_DIR, SEQUENCE_LEN


def profile(
    model_path: Path,
    n_runs: int = 1000,
    device: str = "cpu",
    backend: str = "auto",
) -> dict:
    """Run n_runs single-sample inferences and return latency stats (ms).

    Args:
        model_path: Path to a saved .keras / .h5 / .onnx model file.
        n_runs:     Number of timed forward passes (after 10-run warm-up).
        device:     ``"cpu"`` forces CPU-only inference (reproducible baseline).
                    ``"gpu"`` allows TensorFlow to use an available GPU/MPS device.
        backend:    ``"keras"`` | ``"onnx"`` | ``"auto"`` (default: infer from suffix).
    """
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    suffix = Path(model_path).suffix.lower()
    if backend == "auto":
        backend = "onnx" if suffix == ".onnx" else "keras"

    feature_dim = FEATURE_DIM
    param_count = 0
    if backend == "onnx":
        from backend.api.onnx_runner import OnnxRunner
        model = OnnxRunner(Path(model_path))
        # ONNX input dim may exceed FEATURE_DIM when feature_mode != "raw".
        try:
            import onnxruntime as ort
            sess = ort.InferenceSession(str(model_path))
            shape = sess.get_inputs()[0].shape
            feature_dim = int(shape[-1])
        except Exception:  # noqa: BLE001
            pass
        param_count = model.count_params()
        print(f"Loaded (ONNX): {model_path}  (~{param_count:,} params, feature_dim={feature_dim})")
        predict = model.predict
    else:
        model = tf.keras.models.load_model(str(model_path))
        param_count = int(model.count_params())
        feature_dim = int(model.input_shape[-1])
        print(f"Loaded (Keras): {model_path}  ({param_count:,} params, feature_dim={feature_dim})")
        predict = lambda x: model.predict(x, verbose=0)  # noqa: E731

    dummy = np.zeros((1, SEQUENCE_LEN, feature_dim), dtype=np.float32)

    # Warm-up — bigger for ONNX since it JITs kernels lazily.
    for _ in range(20 if backend == "onnx" else 10):
        predict(dummy)

    latencies_ms = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        predict(dummy)
        latencies_ms.append((time.perf_counter() - t0) * 1000)

    arr = np.array(latencies_ms)
    stats = {
        "n_runs": n_runs,
        "device": device,
        "backend": backend,
        "param_count": param_count,
        "feature_dim": feature_dim,
        "mean_ms":   float(np.mean(arr)),
        "std_ms":    float(np.std(arr)),
        "min_ms":    float(np.min(arr)),
        "p50_ms":    float(np.percentile(arr, 50)),
        "p95_ms":    float(np.percentile(arr, 95)),
        "p99_ms":    float(np.percentile(arr, 99)),
        "max_ms":    float(np.max(arr)),
        "throughput_fps": float(1000.0 / np.mean(arr)),
    }
    return stats


def write_profile_report(stats: dict, model_path: Path, reports_dir: Path, target_ms: float = 30.0) -> Path:
    p95 = stats["p95_ms"]
    status = "PASS ✅" if p95 < target_ms else "FAIL ❌"

    try:
        rel_model = Path(model_path).resolve().relative_to(_REPO_ROOT)
    except ValueError:
        rel_model = model_path

    report = f"""# Inference Latency Profile — {stats.get('backend', 'keras').upper()}

**Model:** `{rel_model}`
**Backend:** {stats.get('backend', 'keras').upper()}
**Device:** {stats.get('device', 'CPU').upper()} (single sample, no batching)
**Runs:** {stats['n_runs']}
**Params:** {stats.get('param_count', 0):,}
**Target p95 latency:** < {target_ms:.0f} ms

## Results

| Stat | Value |
|---|---|
| Mean | {stats['mean_ms']:.2f} ms |
| Std  | {stats['std_ms']:.2f} ms |
| Min  | {stats['min_ms']:.2f} ms |
| p50  | {stats['p50_ms']:.2f} ms |
| **p95**  | **{stats['p95_ms']:.2f} ms** |
| p99  | {stats['p99_ms']:.2f} ms |
| Max  | {stats['max_ms']:.2f} ms |
| Throughput | {stats['throughput_fps']:.1f} samples/sec |

## Verdict

**p95 = {p95:.2f} ms — {status}**
{"Real-time feasible: within the Phase 4 target." if p95 < target_ms else f"Exceeds {target_ms:.0f} ms target — try ONNX export: make export-onnx IN=<path>"}
"""

    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / "inference_profile.md"
    path.write_text(report)
    return path


def export_final(model_path: Path, out_dir: Path) -> Path:
    """Copy/re-save the best checkpoint as lstm_final.keras (Phase 3 handoff)."""
    model = tf.keras.models.load_model(str(model_path))
    final_path = out_dir / "lstm_final.keras"
    model.save(str(final_path))
    return final_path


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Profile SignLearn LSTM inference latency")
    p.add_argument(
        "--model", type=Path,
        default=CHECKPOINTS_DIR / "lstm_best.keras",
    )
    p.add_argument("--n",           type=int,  default=1000, help="Number of timed runs")
    p.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    p.add_argument("--out-dir",     type=Path, default=CHECKPOINTS_DIR)
    p.add_argument(
        "--device",
        choices=["cpu", "gpu"],
        default="cpu",
        help="Device to run inference on. 'cpu' is the reproducible default; "
             "'gpu' enables tensorflow-metal / CUDA for GPU latency measurement.",
    )
    p.add_argument(
        "--backend",
        choices=["auto", "keras", "onnx"],
        default="auto",
        help="Inference backend (default: auto-detect from file extension).",
    )
    p.add_argument(
        "--max-p95-ms",
        type=float,
        default=30.0,
        help="Phase 4 latency target. Exit non-zero if p95 exceeds this. "
             "Set to 0 to disable the gate. (default: 30 ms for ONNX backend; "
             "Keras CPU is typically 100-400 ms — use --max-p95-ms 0 when profiling Keras.)",
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()

    device_label = args.device.upper()
    print(f"Profiling {args.n} single-sample inferences on {device_label} ({args.backend}) …")
    stats = profile(args.model, n_runs=args.n, device=args.device, backend=args.backend)

    report_path = write_profile_report(stats, args.model, args.reports_dir, target_ms=args.max_p95_ms if args.max_p95_ms > 0 else 30.0)
    print(f"\nLatency profile → {report_path}")
    print(f"  mean={stats['mean_ms']:.1f} ms  "
          f"p50={stats['p50_ms']:.1f} ms  "
          f"p95={stats['p95_ms']:.1f} ms  "
          f"p99={stats['p99_ms']:.1f} ms  "
          f"throughput={stats['throughput_fps']:.1f} fps")

    # Only export a final .keras copy if in Keras mode (not useful for ONNX re-export).
    if stats.get("backend") == "keras":
        final_path = export_final(args.model, args.out_dir)
        print(f"\nFinal export    → {final_path}")

    p95 = stats["p95_ms"]
    if args.max_p95_ms > 0 and p95 > args.max_p95_ms:
        print(
            f"\n[FAIL] p95 = {p95:.2f} ms exceeds the Phase 4 target of "
            f"{args.max_p95_ms:.0f} ms.",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"\n[PASS] p95 = {p95:.2f} ms — within target ({args.max_p95_ms:.0f} ms).")
