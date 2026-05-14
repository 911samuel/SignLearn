# SignLearn — Operations Runbook

## Health check

```bash
curl http://127.0.0.1:5001/health
```

Expected response:

```json
{
  "status": "ok",
  "uptime_seconds": 42.1,
  "sequence_len": 30,
  "feature_dim": 126,
  "model_loaded": true,
  "backend": "onnx",
  "num_classes": 36,
  "model_sha256": "a3f9...",
  "load_error": null
}
```

If `model_loaded` is `false`, check `load_error` and inspect server logs:

```bash
tail -f /tmp/signlearn_server.log
```

## Metrics

```bash
# Prometheus text format (for Grafana / Prometheus scrape)
curl http://127.0.0.1:5001/metrics

# JSON (for programmatic consumers)
curl "http://127.0.0.1:5001/metrics?format=json"
```

Key metrics to watch:

| Metric | Alert threshold | Action |
|---|---|---|
| `signlearn_inference_latency_ms` p95 | > 30 ms | Export to ONNX, reduce feature dim |
| `signlearn_no_hand_rate` | > 0.3 (30%) | Check MediaPipe version / lighting |
| `signlearn_class_confidence_rolling` | < 0.5 for any class | Collect more training data for that class |
| `signlearn_predictions_total` / `signlearn_frames_total` | < 0.01 | Model may have stopped responding |

## Hot-swap a new checkpoint

Without dropping WebSocket connections:

```bash
# 1. Export new checkpoint to ONNX
make export-onnx IN=artifacts/runs/new-run/checkpoints/bilstm_best.keras

# 2. Profile it (must pass p95 < 30ms gate)
make profile MODEL=artifacts/runs/new-run/bilstm_best.onnx

# 3. Trigger in-process reload via the admin endpoint
#    Requires SIGNLEARN_ADMIN_TOKEN to be set on the server.
export SIGNLEARN_ADMIN_TOKEN="your-secret-token"
curl -X POST http://127.0.0.1:5001/admin/reload \
     -H "X-Admin-Token: $SIGNLEARN_ADMIN_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"path": "artifacts/runs/new-run/bilstm_best.onnx"}'
```

**Security:** `/admin/reload` is disabled (returns 403) if `SIGNLEARN_ADMIN_TOKEN` env var is not set. Always set this before starting the server in any environment where the endpoint should be usable.

If reload fails (returns 503), the old model stays active. Check `load_error` in `/health`.

## Model not loading (503)

```
{"error": "model_not_ready", "message": "Model has not loaded yet"}
```

1. Check the path in `SIGNLEARN_MODEL_PATH` (or `backend/api/config.py` default).
2. Verify the file exists: `ls -lh artifacts/checkpoints/lstm_best.keras`
3. If Git LFS: `git lfs pull`
4. Try loading manually: `python -c "import tensorflow as tf; tf.keras.models.load_model('...')"`.

## Malformed landmark frames (422)

```
{"error": "landmark_validation_error", "message": "Expected 126 landmark values, got shape (63,)"}
```

This is a client-side bug — the frontend is sending half a frame. Check the MediaPipe landmark flattening logic in `frontend/src/hooks/useSignRecognition.ts`.

## High no-hand-frame rate

If `/metrics` shows `no_hand_rate` > 30%:

1. Ensure the camera is not blocked or the room is not too dark.
2. Check MediaPipe model version — `models/hand_landmarker.task` must match the Tasks API version.
3. Add logging to the frontend WebSocket handler to count frames where `landmarks` is empty.

## Prediction confidence stuck low

If `class_confidence_rolling` stays below 0.5 for multiple classes:

1. Run `make audit` — check per-class sample counts.
2. If a class has < 40 samples, record more: `make record LABEL=<class>`.
3. Retrain: `make train ARCH=bilstm FEATURE_MODE=engineered RUN_NAME=bilstm-retrain-v2`.
4. Export + hot-swap.

## Training a new model from scratch

```bash
# 1. Collect data
make audit                             # check current coverage
make record LABEL=hello                # record missing classes

# 2. Train (use sweep for production quality)
make sweep SWEEP_CFG=configs/sweeps/phase3.yaml

# 3. Identify best run
cat artifacts/reports/sweeps/phase3.md

# 4. Export best checkpoint
make export-onnx IN=artifacts/runs/<winner>/checkpoints/<arch>_best.keras

# 5. Profile
make profile MODEL=artifacts/runs/<winner>/<arch>_best.onnx

# 6. Hot-swap (see above)
```

## Backup and recovery

Checkpoints are tracked via Git LFS:

```bash
git add artifacts/runs/new-run/checkpoints/bilstm_best.keras
git commit -m "feat: add bilstm-eng-v2 checkpoint"
git push
```

To restore a previous checkpoint:

```bash
git log --oneline artifacts/
git checkout <sha> -- artifacts/runs/<run>/checkpoints/<arch>_best.keras
```

## Logs

| Log | Location | Contents |
|---|---|---|
| Server | stdout / `nohup.out` | Flask + SocketIO events |
| Training | `artifacts/runs/<run>/logs/` | TensorBoard event files |
| Dataset audit | `artifacts/reports/dataset_audit.md` | Per-class stats, Gini, dedup |
| Signer leakage | `artifacts/reports/audit_signers.md` | Cross-split signer overlap |
| Inference profile | `artifacts/reports/inference_profile.md` | p50/p95/p99 latency |
| Sweep results | `artifacts/reports/sweeps/<id>.md` | Arch × feature comparison table |
