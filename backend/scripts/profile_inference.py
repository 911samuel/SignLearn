"""Phase 2 — Subtask 7: Inference latency profiling and final model export.

Runs 1000 single-sample forward passes (CPU, no batching) and reports
mean / p50 / p95 / p99 latency + throughput. Target: p95 < 500 ms.

Writes:
  artifacts/reports/inference_profile.md
  artifacts/checkpoints/lstm_final.keras   (canonical Phase 3 handoff)

Usage
-----
python scripts/profile_inference.py
python scripts/profile_inference.py --model artifacts/checkpoints/lstm_best.keras --n 500
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from backend.model.config import CHECKPOINTS_DIR, FEATURE_DIM, REPORTS_DIR, SEQUENCE_LEN


def profile(model_path: Path, n_runs: int = 1000, device: str = "cpu") -> dict:
    """Run n_runs single-sample inferences and return latency stats (ms).

    Args:
        model_path: Path to a saved .keras or .h5 model file.
        n_runs:     Number of timed forward passes (after 10-run warm-up).
        device:     ``"cpu"`` forces CPU-only inference (reproducible baseline).
                    ``"gpu"`` allows TensorFlow to use an available GPU/MPS device.
    """
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    model = tf.keras.models.load_model(str(model_path))
    print(f"Loaded: {model_path}  ({model.count_params():,} params)")

    dummy = np.zeros((1, SEQUENCE_LEN, FEATURE_DIM), dtype=np.float32)

    # Warm-up: 10 passes so TF graph is compiled before we time
    for _ in range(10):
        model.predict(dummy, verbose=0)

    latencies_ms = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        model.predict(dummy, verbose=0)
        latencies_ms.append((time.perf_counter() - t0) * 1000)

    arr = np.array(latencies_ms)
    stats = {
        "n_runs": n_runs,
        "device": device,
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


def write_profile_report(stats: dict, model_path: Path, reports_dir: Path) -> Path:
    target_ms = 500.0
    p95 = stats["p95_ms"]
    status = "PASS ✅" if p95 < target_ms else "FAIL ❌"

    try:
        rel_model = Path(model_path).resolve().relative_to(_REPO_ROOT)
    except ValueError:
        rel_model = model_path

    report = f"""# Phase 2 — Inference Latency Profile

**Model:** `{rel_model}`
**Device:** {stats.get('device', 'CPU').upper()} (single sample, no batching)
**Runs:** {stats['n_runs']}
**Target p95 latency:** < {target_ms:.0f} ms

## Results

| Stat | Value |
|---|---|
| Mean | {stats['mean_ms']:.1f} ms |
| Std  | {stats['std_ms']:.1f} ms |
| Min  | {stats['min_ms']:.1f} ms |
| p50  | {stats['p50_ms']:.1f} ms |
| p95  | {stats['p95_ms']:.1f} ms |
| p99  | {stats['p99_ms']:.1f} ms |
| Max  | {stats['max_ms']:.1f} ms |
| Throughput | {stats['throughput_fps']:.1f} samples/sec |

## Verdict

**p95 = {p95:.1f} ms — {status}**
{"Real-time feasible: well within the 500 ms Phase 2 target." if p95 < target_ms else "Exceeds target — optimisation required before Phase 3 integration."}
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
        "--max-p95-ms",
        type=float,
        default=500.0,
        help="Phase 2 latency target. Exit non-zero if p95 exceeds this. "
             "Set to 0 to disable the gate.",
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()

    device_label = args.device.upper()
    print(f"Profiling {args.n} single-sample inferences on {device_label} …")
    stats = profile(args.model, n_runs=args.n, device=args.device)

    report_path = write_profile_report(stats, args.model, args.reports_dir)
    print(f"\nLatency profile → {report_path}")
    print(f"  mean={stats['mean_ms']:.1f} ms  "
          f"p50={stats['p50_ms']:.1f} ms  "
          f"p95={stats['p95_ms']:.1f} ms  "
          f"p99={stats['p99_ms']:.1f} ms  "
          f"throughput={stats['throughput_fps']:.1f} fps")

    final_path = export_final(args.model, args.out_dir)
    print(f"\nFinal export    → {final_path}  (Phase 3 handoff)")

    p95 = stats["p95_ms"]
    if args.max_p95_ms > 0 and p95 > args.max_p95_ms:
        print(
            f"\n[FAIL] p95 = {p95:.1f} ms exceeds the Phase 2 target of "
            f"{args.max_p95_ms:.0f} ms.",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"\n[PASS] p95 = {p95:.1f} ms — within Phase 2 target ({args.max_p95_ms:.0f} ms).")
