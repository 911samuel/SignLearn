# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

SignLearn is a real-time American Sign Language (ASL) recognition system. MediaPipe Hands extracts landmarks in-browser; a sequence classifier on the backend predicts signs and logs a transcript for hearing users.

**Member A** (ML/AI Lead): data pipeline, model training, MediaPipe extraction — branch `feature/model-training`.  
**Member B** (Full-Stack + Frontend Lead): Flask backend, WebSocket, React UI, speech-to-text — branch `dev-web`.

Vocabulary: 93 classes — `a`–`z` (26), `0`–`9` (10), 24 static words, 33 dynamic words (snake_case). Full list: `docs/vocabulary.md`.  
Currently trained on **36 classes** (26 letters + 10 digits, ~62K augmented sequences). **Best run: TCN+raw lr=5e-4, 97.84% test accuracy** (`artifacts/runs/phase3-raw-balanced__arch=tcn_fm=raw_lr=0.0005_do=0.4/`). Several other architectures (BiLSTM-v1, CNN-BiLSTM) also above 96%. Phase 3 sweep complete. The residual ~2% error is dominated by linguistically-equivalent class pairs (`two`/`v`, `six`/`w`, `zero`/`o`) which share identical ASL handshapes. ONNX export of the TCN winner is the production checkpoint (`artifacts/checkpoints/tcn_best.onnx`, p95 = 5.6ms, 196 fps). See `artifacts/reports/` for current benchmarks.

**Word model (research extension):** 61-class BiLSTM trained on WLASL data, `artifacts/runs/word-bilstm-v1/` — 26% top-1, 62% top-5 test accuracy. Vocabulary limited by WLASL data scarcity; not currently served. Intended future work: replace WLASL clips with native-signer recordings of conversational vocabulary.

---

## Commands

```bash
# One-command setup (pip install + label map + MediaPipe model)
make setup

# Individual setup steps
pip install -r requirements.txt
python -m backend.data.label_map
make model  # downloads MediaPipe hand landmarker

# Download Kaggle ASL datasets (requires ~/.kaggle/kaggle.json)
python backend/scripts/download_datasets.py --dataset alphabet
python backend/scripts/download_datasets.py --dataset digits

# Augment minority classes to target sample count
make augment TARGET=600

# Record a labelled sequence from webcam (with diversity metadata)
make record LABEL=hello
# or with full diversity prompts (angle × lighting)
python backend/scripts/record_vocabulary.py --words hello --diversity-matrix

# Audit dataset quality
make audit   # writes artifacts/reports/dataset_audit.md + audit_signers.md

# Train a single run
make train ARCH=bilstm FEATURE_MODE=raw RUN_NAME=bilstm-raw-v1 EPOCHS=100
# or directly
python backend/scripts/train_model.py --arch tcn --feature-mode raw --run-name tcn-raw-v1

# Hyperparameter sweep (YAML-driven, unattended)
make sweep SWEEP_CFG=configs/sweeps/phase3_raw_smoke.yaml

# Compare / evaluate runs
python backend/scripts/evaluate_model.py --runs bilstm-v1 tcn-v1 cnn_bilstm-v1
make compare  # evaluates all runs in artifacts/runs/

# Export best checkpoint to ONNX (for <30ms CPU inference)
make export-onnx IN=artifacts/runs/bilstm-raw-v1/checkpoints/bilstm_best.keras

# Profile inference latency (Keras or ONNX)
make profile MODEL=artifacts/runs/bilstm-raw-v1/bilstm_best.onnx
python backend/scripts/profile_inference.py --backend keras --max-p95-ms 0 \
    --model artifacts/runs/bilstm-raw-v1/checkpoints/bilstm_best.keras

# Run all tests (requires env vars)
SIGNLEARN_ASYNC_MODE=threading SIGNLEARN_SECRET_KEY=test-secret pytest
make test

# Run a single test file
make test-file FILE=tests/test_augment.py

# Start the backend server (http://127.0.0.1:5001)
make serve

# Start the frontend dev server (http://localhost:5173)
make frontend

# Hot-swap model checkpoint without dropping connections (requires SIGNLEARN_ADMIN_TOKEN)
curl -X POST http://127.0.0.1:5001/admin/reload \
     -H "X-Admin-Token: $SIGNLEARN_ADMIN_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"path": "artifacts/runs/new-run/checkpoints/bilstm_best.keras"}'
```

**Notes:**
- `SIGNLEARN_ASYNC_MODE=threading` is required by all tests — Flask-SocketIO eventlet mode conflicts with pytest threads.
- `SIGNLEARN_SECRET_KEY` is required by route/socket tests.
- `models/hand_landmarker.task` is gitignored; every developer must download it.
- Scripts use `mediapipe.tasks` (Tasks API) — not the legacy `mediapipe.python.solutions` API (unavailable on macOS ARM in mediapipe ≥ 0.10).

---

## Architecture

### End-to-end data flow

```
Browser webcam
  → MediaPipe Hands (in-browser, frontend/src/hooks/useSignRecognition.ts)
  → WebSocket "frame" event  {landmarks: [126 floats], t: unix_ms}
  → backend/api/socket_handlers.py  →  FrameBuffer.push()
  → backend/api/model_loader.run_inference_probs()
  → backend/api/smoothing.PredictionSmoother.update()  (EMA + confidence gate)
  → WebSocket "prediction" event  {label, confidence, ready}
  → frontend HearingPanel.tsx
```

### ML pipeline (`backend/`)

**Landmark shape:** 2 hands × 21 landmarks × 3 coords = `(30, 126)` float32 per sequence. Left hand occupies `[:63]`, right hand `[63:]`; absent hand slots are zero-padded. The wrist (index 0) is centred at origin and max inter-landmark distance is unit-scaled per hand (`backend/data/normalize.py`).

**Model architectures** — all six are registered in `backend/model/architectures/__init__.py`:

| Name | File | Notes |
|---|---|---|
| `lstm` | `architectures/lstm.py` | Stacked LSTM 128→64 |
| `bilstm` | `architectures/bilstm.py` | BiLSTM 128→64; **83.5% test acc on 36 cls (raw, 100 epochs)** |
| `transformer` | `architectures/transformer.py` | 2-layer encoder, 4 heads, d_model=128; best on digits |
| `tcn` | `architectures/tcn.py` | Dilated Conv1D stack (dilations 1,2,4,8), residual blocks |
| `cnn_bilstm` | `architectures/cnn_bilstm.py` | Conv1D×2 front-end → BiLSTM→BiLSTM |
| `conformer_lite` | `architectures/conformer_lite.py` | 2 blocks Conv1D + MHA(4 heads) + FFN |

Select with `--arch` flag; config is persisted to `artifacts/runs/<run-name>/reports/config.json`.

**Feature modes** (`backend/data/features.py`) — set via `TrainConfig.feature_mode`:
- `raw` (default): plain normalized `(T, 126)`
- `raw+velocity`: appends per-frame Δ → `(T, 252)`
- `raw+velocity+angles`: also appends 5 joint angles/hand → `(T, 262)`
- `raw+velocity+acceleration`: second-order diff added → `(T, 378)`
- `engineered`: velocity + acceleration + pairwise distances + angles → `(T, 398)`

**Important:** `engineered` and acceleration modes add near-zero features for static images (all 30 frames identical → velocity/acceleration ≈ 0). Prefer `raw` or `raw+velocity` for the current Kaggle-sourced dataset.

**Label map:** `artifacts/label_map.json` is the canonical index (0–92). Generated from `docs/vocabulary.md`. Digit directories on disk (`0`–`9`) are aliased to word form (`zero`–`nine`) by `backend/data/label_map.resolve_label`. Training uses a *compact* re-index over only the classes present in `data/processed/train/` — see `compact_label_map()` in `backend/model/config.py`.

**Augmentation** (`backend/data/augment.py`): default `random_augment` uses mild probs (rotate, scale, translate, noise, drop). Pass `probs=TRAINING_PROBS` (exported from the same module) to enable the aggressive training profile that also includes `rotate3d` and `speed_warp`. Horizontal flip defaults to 0 — handedness is linguistically meaningful.

**Artifact layout:**
```
artifacts/
├── checkpoints/          # legacy flat path; lstm_best.keras / lstm_final.keras
├── runs/<run-name>/
│   ├── checkpoints/      # <arch>_best.keras, <arch>_final.keras
│   ├── reports/          # config.json, history.json, metrics.json,
│   │                     # classification_report.txt, confusion_matrix.png
│   └── logs/             # TensorBoard
├── reports/
│   ├── dataset_audit.md / .json
│   ├── model_comparison.md
│   ├── data_collection_recommendation.md
│   └── inference_profile.md
└── label_map.json
```

### Backend API (`backend/api/`)

Flask + Flask-SocketIO in `threading` mode. One `FrameBuffer` per connected signer (keyed by `request.sid`). The buffer normalizes frames, accumulates 30, runs inference, and passes the full softmax vector through `PredictionSmoother` before emitting.

**`PredictionSmoother` knobs** (`backend/api/smoothing.py`):
- `ema_alpha=0.6` — weight on the newest frame's probabilities
- `conf_threshold=0.75` — below this, emit `{label: null}`
- `repeat_cooldown_frames=15` — suppress re-emitting the same label
- `stride=1` — predict every N frames

The server's default checkpoint path is `artifacts/checkpoints/lstm_best.keras` (`backend/api/config.py`). Swap to the Transformer by symlinking/copying `artifacts/runs/tx-v1/checkpoints/transformer_best.keras` there.

**WebSocket wire format (unchanged):**
```
client → server:  emit("frame",  {"landmarks": [126 floats], "t": unix_ms})
server → client:  emit("prediction", {"label": str|null, "confidence": float|null, "ready": bool})
client → server:  emit("reset")
server → client:  emit("reset_ack", {})
```

Normalization (`normalize_frame`) runs on the backend — frontend sends **raw** landmark values.

**Inference backend** (`backend/api/model_loader.py`): `ModelHolder` singleton loads `.keras` (Keras) or `.onnx` (OnnxRunner) based on extension, exposes `reload(path)` for atomic hot-swap under `RLock`. Module-level shims `run_inference_probs`, `get_class_names`, `is_loaded` keep call sites unchanged.

**Error handling** (`backend/api/errors.py`): typed hierarchy `SignLearnError` → `LandmarkValidationError` (422) / `ModelNotReadyError` (503). `register_error_handlers(app)` converts them to structured JSON. Socket handlers catch these and emit `error` / `prediction` events accordingly.

**Telemetry** (`backend/api/telemetry.py`): `METRICS` singleton — `record_prediction`, `record_no_hand_frame`. Exposed via `GET /metrics?format=prometheus` (Prometheus text) or `?format=json`.

**REST endpoints:**

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Server uptime + model SHA256 + backend type |
| `GET` | `/metrics` | Prometheus metrics (add `?format=json` for JSON) |
| `POST` | `/admin/reload` | Hot-swap checkpoint; requires `X-Admin-Token` header |
| `POST` | `/rooms` | Allocate room code |
| `GET` | `/rooms/<id>` | Room state |
| `GET` | `/transcript?room_id=X` | Conversation log |
| `POST` | `/feedback` | User feedback |
| `POST` | `/corrections` | Signer correction of prediction |
| `DELETE` | `/transcript?confirm=1` | Clear transcript (dev only) |

### Frontend (`frontend/`)

React + Next.js 15 App Router. MediaPipe Hands runs **in-browser** (lower latency than backend extraction). Dual-panel layout: left (signer + webcam overlay), right (hearing user + speech input).

Key hooks: `src/hooks/useSignRecognition.ts`, `src/hooks/useSpeechToText.ts`.  
Key components: `src/components/SignerPanel.tsx`, `src/components/HearingPanel.tsx`.  
Backend URL: `VITE_BACKEND_URL` (defaults to `http://127.0.0.1:5001`).

---

## Key constraints

- **No CNN stage.** MediaPipe replaces spatial feature extraction; the sequence model handles temporal modeling. This keeps inference on CPU at ≥30 FPS.
- **Compact label remapping is mandatory.** Model output has `num_classes` = number of classes *present in the train split*, not 93. `_remap_labels()` in `train_model.py` converts full-vocab indices to compact indices before computing the loss.
- **Static images lack temporal motion.** Sequences extracted from single images (all frames identical) are valid for static signs only. Dynamic words require real video capture via `extract_landmarks.py`.
- **`artifacts/` is gitignored** except `.keras` files tracked via Git LFS. Always run `git lfs pull` after checkout if you need checkpoints.
- **k/v and r/u confusions are dataset-intrinsic.** In the Kaggle synthetic dataset both `k` and `v` have `|index_z - middle_z| = 0` — only real webcam recordings at different angles can fix this. Same applies to `r`/`u` and the semantic equivalences `two≡v`, `six≡w`.
- **Admin hot-swap requires `SIGNLEARN_ADMIN_TOKEN` env var.** If unset, `/admin/reload` returns 403.

---

## CI

GitHub Actions (`.github/workflows/ci.yml`) — triggers on push/PR to `dev-ml`, `dev-web`, `main`:
- **test**: Python 3.11, `pytest` (LFS checkout included)
- **frontend**: Node 20, `npm ci && npm run build && npm test`
