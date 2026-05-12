# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

SignLearn is a real-time American Sign Language (ASL) recognition system. MediaPipe Hands extracts landmarks in-browser; a sequence classifier on the backend predicts signs and logs a transcript for hearing users.

**Member A** (ML/AI Lead): data pipeline, model training, MediaPipe extraction — branch `feature/model-training`.  
**Member B** (Full-Stack + Frontend Lead): Flask backend, WebSocket, React UI, speech-to-text — branch `dev-web`.

Vocabulary: 93 classes — `a`–`z` (26), `0`–`9` (10), 24 static words, 33 dynamic words (snake_case). Full list: `docs/vocabulary.md`.  
Currently trained on **digits only** (10 classes, 1724 sequences). Remaining 83 classes need recorded data — see `artifacts/reports/data_collection_recommendation.md`.

---

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Build label map from docs/vocabulary.md (required once)
python -m backend.data.label_map

# Download MediaPipe hand landmarker (required once, stored in models/)
curl -L https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task \
     -o models/hand_landmarker.task

# Download Kaggle ASL datasets (requires ~/.kaggle/kaggle.json)
python backend/scripts/download_datasets.py --dataset alphabet
python backend/scripts/download_datasets.py --dataset digits

# Record a landmark sequence from webcam (30 frames → .npy)
python backend/scripts/extract_landmarks.py --out data/processed/train/<label>/<label>_s01_0000.npy

# Audit dataset quality
python backend/scripts/audit_dataset.py

# Train a model (outputs to artifacts/runs/<run-name>/)
python backend/scripts/train_model.py --arch transformer --run-name tx-v1 --epochs 80
python backend/scripts/train_model.py --arch bilstm     --run-name bilstm-v1
python backend/scripts/train_model.py --arch lstm                            # legacy default

# Compare multiple trained runs
python backend/scripts/evaluate_model.py --runs lstm-v1 bilstm-v1 tx-v1

# Evaluate a single checkpoint
python backend/scripts/evaluate_model.py --model artifacts/runs/tx-v1/checkpoints/transformer_best.keras

# Profile inference latency
python backend/scripts/profile_inference.py --model artifacts/runs/tx-v1/checkpoints/transformer_best.keras

# Run all tests (requires env vars)
SIGNLEARN_ASYNC_MODE=threading SIGNLEARN_SECRET_KEY=test-secret pytest

# Run a single test file
SIGNLEARN_ASYNC_MODE=threading SIGNLEARN_SECRET_KEY=test-secret pytest tests/test_augment.py -v

# Start the backend server (http://127.0.0.1:5001)
python backend/scripts/run_server.py

# Start the frontend dev server (http://localhost:5173)
cd frontend && npm install && npm run dev
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

**Model architectures** — all three are registered in `backend/model/architectures/__init__.py`:

| Name | File | Notes |
|---|---|---|
| `lstm` | `architectures/lstm.py` | Stacked LSTM 128→64, Phase 2 baseline |
| `bilstm` | `architectures/bilstm.py` | BiLSTM 128→64 |
| `transformer` | `architectures/transformer.py` | 2-layer encoder, 4 heads, d_model=128; **best performer (98.4% test acc on digits)** |

Select with `--arch` flag; config is persisted to `artifacts/runs/<run-name>/reports/config.json`.

**Feature modes** (`backend/data/features.py`) — set via `TrainConfig.feature_mode`:
- `raw` (default): plain normalized `(T, 126)`
- `raw+velocity`: appends per-frame Δ → `(T, 252)`
- `raw+velocity+angles`: also appends 5 finger joint angles per hand → `(T, 262)`

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

**REST endpoints:**

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Server + model load status |
| `POST` | `/speech-to-text` | Log browser STT result `{"text":"..."}` |
| `GET` | `/transcript?limit=100` | Session conversation log |
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

---

## CI

GitHub Actions (`.github/workflows/ci.yml`) — triggers on push/PR to `dev-ml`, `dev-web`, `main`:
- **test**: Python 3.11, `pytest` (LFS checkout included)
- **frontend**: Node 20, `npm ci && npm run build && npm test`
