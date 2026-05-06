# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

SignLearn is a real-time American Sign Language (ASL) recognition system. It uses MediaPipe Hands for landmark extraction and a stacked LSTM model for sequence classification across 93 vocabulary classes (26 letters, 10 digits, 24 static words, 33 dynamic words). See `docs/vocabulary.md` for the full list.

**Member A** (this context): ML/AI Lead — data pipeline, model design, training, MediaPipe extraction.  
**Member B**: Full-Stack + Frontend Lead — Flask backend, WebSocket, React UI, speech-to-text.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Download MediaPipe hand landmarker model (required once, stored in models/)
curl -L https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task \
     -o models/hand_landmarker.task

# Verify webcam + MediaPipe are working (smoke test, appends result to docs/hardware.md)
python scripts/test_mediapipe.py

# Collect a landmark sequence sample (30 frames → .npy)
python scripts/extract_landmarks.py --out data/processed/<label>/sample_001.npy --frames 30

# Download ASL datasets from Kaggle (requires ~/.kaggle/kaggle.json)
python scripts/download_datasets.py --dataset alphabet
python scripts/download_datasets.py --dataset digits

# Verify raw dataset image integrity
python backend/data_verification.py data/raw/digits

# Run all tests
pytest

# Run a specific test file verbosely
pytest tests/test_pipeline.py -v
```

Notes:
- `tests/test_pipeline.py` requires `data/processed/sample.npy` (run `extract_landmarks.py` first).
- Scripts use the **MediaPipe Tasks API** (`mediapipe.tasks`) — not the legacy `mediapipe.python.solutions` which is unavailable on macOS ARM in mediapipe ≥ 0.10.
- `models/` is gitignored; `hand_landmarker.task` must be downloaded locally by each developer.

## Architecture

### ML Pipeline

**Input → Landmark extraction → Sequence → LSTM → Prediction**

1. Webcam frame → MediaPipe Hands → up to 2 hands × 21 landmarks × 3 coords = 126 floats per frame (63 per hand, both hands concatenated; missing hand zero-padded)
2. 30 consecutive frames → shape `(30, 126)` float32 array (zero-padded if no hand detected)
3. Two-layer stacked LSTM (128 → 64 units) with Masking → softmax over 93 classes
4. Target: ≥85% validation accuracy, <500ms inference latency (p95)

Key decision: **No CNN stage.** MediaPipe replaces spatial feature extraction; LSTM handles temporal modeling. This allows real-time CPU inference at ≥30 FPS.

### Data

- `data/raw/` — static image datasets (ASL Alphabet ~3000 imgs/class, Digits ~100 imgs/class)
- `data/processed/` — landmark sequences as `.npy` files in shape `(30, 126)`, float32
- `data/external/` — third-party metadata / WLASL subsets
- `models/` — trained `.h5` model files (gitignored)

**Training data strategy**: Public datasets are static images and have bias issues. Primary approach is custom self-recorded data via `extract_landmarks.py` targeting 50 samples per class (93 × 50 ≈ 4650 total sequences).

### Vocabulary

93 classes: `a`–`z` (26), `0`–`9` (10), 24 static words, 33 dynamic words (all snake_case). Full list in `docs/vocabulary.md`.

### Backend (Phase 3 — complete)

Flask + Flask-SocketIO server (`threading` async mode). Landmark frames arrive over WebSocket, a 30-frame sliding window triggers LSTM inference, confident predictions are emitted back and logged to SQLite.

**Running the backend:**

```bash
# Install backend deps (flask, flask-socketio, flask-cors, websocket-client)
pip install -r requirements.txt

# Start the dev server on http://127.0.0.1:5001
python scripts/run_server.py

# End-to-end smoke test (starts its own server subprocess)
python scripts/e2e_smoke.py

# WebSocket latency profiler (p95 target < 500 ms)
python scripts/profile_ws.py
```

**REST endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Server + model status |
| `POST` | `/speech-to-text` | Log browser STT result `{"text": "..."}` |
| `GET` | `/transcript?limit=100` | Full session conversation log |
| `DELETE` | `/transcript?confirm=1` | Clear transcript (dev only) |

**WebSocket events (Member B wire format):**

```
client → server:  emit("frame",  {"landmarks": [126 floats], "t": <unix ms>})
server → client:  emit("prediction", {"label": str|null, "confidence": float|null, "ready": bool})
client → server:  emit("reset")          # clear sliding window
server → client:  emit("reset_ack", {})
```

Normalization (`normalize_frame`) runs on the backend — frontend sends raw landmark values.

### Frontend (planned, Phase 4)

React app. MediaPipe Hands runs **in-browser** (lower latency than backend extraction). Dual-panel layout: left (signer + webcam), right (hearing user + speech input). `useSignRecognition` custom hook.

## Phases (14-week plan)

| Phase | Weeks | Goal |
|-------|-------|------|
| 0 | 1 | Environment setup |
| 1 | 2–3 | Data pipeline & landmark preprocessing |
| 2 | 4–6 | LSTM model training & evaluation |
| 3 | 5–7 | Backend API & WebSocket server |
| 4 | 6–9 | React frontend |
| 5 | 10 | Full system integration |
| 6 | 11–12 | Testing, evaluation, usability |
| 7 | 13–14 | Documentation & defense prep |

Full plan: `docs/sign_learn.md`

## CI

GitHub Actions (`.github/workflows/ci.yml`) runs `pytest` on push/PR to `dev-ml` and `main` branches using Python 3.11 on ubuntu-latest.

## Key Docs

- `docs/sign_learn.md` — full 14-week implementation plan
- `docs/vocabulary.md` — 75-class label list (ML-ready snake_case)
- `docs/data_gaps.md` — dataset limitations and fallback strategies
- `docs/hardware.md` — M2 Pro GPU/CPU strategy (tensorflow-metal)
- `docs/research_notes.md` — MediaPipe + LSTM architecture justification
