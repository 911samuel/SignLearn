# Phase 2 — Inference Latency Profile

**Model:** `artifacts/checkpoints/lstm_best.keras`
**Device:** CPU (single sample, no batching)
**Runs:** 1000
**Target p95 latency:** < 500 ms

## Results

| Stat | Value |
|---|---|
| Mean | 31.9 ms |
| Std  | 5.2 ms |
| Min  | 27.5 ms |
| p50  | 31.1 ms |
| p95  | 36.4 ms |
| p99  | 63.2 ms |
| Max  | 84.3 ms |
| Throughput | 31.3 samples/sec |

## Verdict

**p95 = 36.4 ms — PASS ✅**
Real-time feasible: well within the 500 ms Phase 2 target.
