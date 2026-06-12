# SignLearn — Project Report

**Branch:** `feature/model-training`  
**Member A (ML/AI Lead):** Samuel Abayizera  
**Date:** May 2026

---

## 1. Project Overview

SignLearn is a real-time American Sign Language (ASL) recognition system that bridges communication between deaf/hard-of-hearing signers and hearing users. The system uses **MediaPipe Hands** running in-browser to extract hand landmarks, streams the landmark data over WebSocket to a Flask backend, and classifies the sign using a deep learning sequence model.

This report covers the full ML pipeline: dataset preparation, architecture experiments, hyperparameter sweep, ONNX export, and final production deployment.

---

## 2. System Architecture

```
Browser (Next.js 15)
  └─ MediaPipe Hands (in-browser, GPU delegate)
       └─ 126 floats/frame (2 hands × 21 landmarks × 3 coords)
            └─ WebSocket "frame" event
                 └─ Flask + SocketIO backend
                      └─ FrameBuffer (30-frame sliding window)
                           └─ TCN model (ONNX, CPU)
                                └─ PredictionSmoother (EMA + confidence gate)
                                     └─ WebSocket "prediction" event
                                          └─ HearingPanel (captions)
```

**Key design decisions:**
- MediaPipe runs **in-browser** — no raw video leaves the device; only 126 numbers per frame are sent to the server.
- The backend receives raw (un-normalized) landmarks and normalizes them server-side (wrist-centred, unit-scaled per hand).
- ONNX runtime serves the model at sub-millisecond latency on CPU — no GPU required in production.

---

## 3. Dataset

| Property | Value |
|---|---|
| Source | Kaggle ASL Alphabet + Digits datasets |
| Classes trained | 36 (letters `a`–`z` + digits `zero`–`nine`) |
| Full vocabulary target | 93 classes |
| Total sequences | **66,666** |
| Train split | 43,657 |
| Validation split | 11,592 |
| Test split | 11,417 |
| Sequence shape | `(30, 126)` float32 |
| Augmentation | rotate, scale, translate, noise, drop, rotate3d, speed_warp |

**How sequences are built:** Each Kaggle image is duplicated 30 times to form a static-pose sequence. This means velocity/acceleration features are near-zero — raw landmark features carry all the signal for this dataset.

**Landmark representation:** Left hand occupies indices `[:63]`, right hand `[63:]`. Absent hands are zero-padded. Wrist (landmark 0) is centred at origin; all landmarks are scaled so the maximum inter-landmark distance per hand = 1.

---

## 4. Model Architecture — TCN (Temporal Convolutional Network)

The winning architecture is a **dilated 1D convolutional stack** with residual connections.

```
Input: (30, 126)
  └─ Conv1D(64, k=1) — projection to feature space
       └─ Dilated residual block (dilation=1)
            Conv1D(64, k=3, dilation=1) → LayerNorm → ReLU → Dropout(0.4)
            Conv1D(64, k=3, dilation=1) → LayerNorm → ReLU → Dropout(0.4)
            + residual
       └─ Dilated residual block (dilation=2)
       └─ Dilated residual block (dilation=4)
       └─ Dilated residual block (dilation=8)
  └─ GlobalAveragePooling1D
  └─ Dense(128, ReLU) → Dropout(0.4)
  └─ Dense(36, Softmax)
```

**Why TCN beat BiLSTM:** Dilated convolutions have a receptive field of 30 frames (matching the window size exactly) while being fully parallelisable during training. BiLSTM processes frames sequentially — slower to train and less efficient at capturing the global pose context that distinguishes similar signs.

**Parameters:** 126,372 (3.4× fewer than the best BiLSTM at 436,068)

---

## 5. Training

### 5.1 Hyperparameter Sweep

A systematic grid sweep was run across 4 architectures × 2 feature modes × 2 learning rates (16 runs total, 100 epochs each):

```
arch:          [bilstm, tcn, cnn_bilstm, conformer_lite]
feature_mode:  [raw, raw+velocity]
learning_rate: [1e-3, 5e-4]
dropout:       [0.4]
batch_size:    32
optimizer:     Adam with ReduceLROnPlateau (factor=0.5, patience=8)
early_stopping: patience=15, restore_best_weights=True
class_weights:  inverse-frequency, capped at 5×
```

### 5.2 Full Sweep Results (all 16 runs)

| Arch | Features | LR | Val acc | Test acc | F1 |
|---|---|---|---:|---:|---:|
| **TCN** | **raw** | **5e-4** | **97.53%** | **97.84%** | **95.5%** |
| TCN | raw | 1e-3 | 97.29% | 97.49% | 94.4% |
| TCN | raw+velocity | 1e-3 | 96.12% | 96.45% | 91.9% |
| CNN-BiLSTM | raw+velocity | 5e-4 | 96.50% | 96.80% | 92.6% |
| CNN-BiLSTM | raw | 5e-4 | 96.28% | 96.76% | 92.8% |
| CNN-BiLSTM | raw | 1e-3 | 95.52% | — | — |
| CNN-BiLSTM | raw+velocity | 1e-3 | 94.00% | — | — |
| TCN | raw+velocity | 5e-4 | 89.20% | — | — |
| BiLSTM | raw | 5e-4 | 88.09% | — | — |
| BiLSTM | raw+velocity | 1e-3 | 87.82% | — | — |
| BiLSTM | raw | 1e-3 | 71.31% | — | — |
| Conformer-Lite | raw | 1e-3 | 74.91% | — | — |
| Conformer-Lite | raw | 5e-4 | 73.45% | — | — |
| Conformer-Lite | raw+velocity | 1e-3 | 72.32% | — | — |
| Conformer-Lite | raw+velocity | 5e-4 | ~72% | — | — |
| BiLSTM | raw+velocity | 5e-4 | 45.92% | — | — |

### 5.3 Progression from Baseline

| Milestone | Model | Test acc | Notes |
|---|---|---|---|
| Phase 1 baseline | LSTM | 73.4% | Initial model |
| Phase 2 baseline | BiLSTM | 84.1% | +10.7pp |
| Phase 3 sweep | TCN | **97.84%** | +13.7pp |

---

## 6. Final Model Performance

**Model:** TCN · raw features · lr=5e-4 · dropout=0.4  
**Checkpoint:** `artifacts/checkpoints/tcn_best.onnx`

| Metric | Value | Target | Status |
|---|---|---|---|
| Test accuracy | **97.84%** | ≥ 95% | ✅ |
| Macro F1 | **95.45%** | ≥ 92% | ✅ |
| Macro precision | 96.25% | — | — |
| Macro recall | 95.32% | — | — |
| ONNX p95 latency | **0.23 ms** | < 30 ms | ✅ |
| Throughput | **4,890 fps** | — | — |
| Parameters | 126,372 | — | — |

### 6.1 Per-class Highlights (winner, test set)

**Strong classes (F1 ≥ 0.98):** a, b, d, e, f, g, h, i, j, l, p, q, s, t, x, y, z, three, five, seven, eight, nine

**Weaker classes:**
| Class | F1 | Root cause |
|---|---|---|
| `two` | 0.49 | ASL semantic equivalence: digit `2` = letter `v` (same handshape) |
| `six` | 0.10 | ASL semantic equivalence: digit `6` = letter `w` (same handshape) |
| `zero` | 0.42 | Confusion with `o` — visually similar in Kaggle synthetic images |

### 6.2 Known Limitations

| Issue | Root cause | Fix |
|---|---|---|
| `two` ↔ `v` (100% overlap) | Same ASL handshape — linguistically equivalent | Context/language model needed |
| `six` ↔ `w` (100% overlap) | Same ASL handshape — linguistically equivalent | Context/language model needed |
| `k` ↔ `v`, `r` ↔ `u` (residual) | Kaggle synthetic: `\|index_z − middle_z\| = 0` | Record real webcam video with depth variation |

The `two`/`six` ambiguities are not model failures — these signs genuinely use the same handshape in ASL. Disambiguation requires sentence-level context.

---

## 7. ONNX Export & Inference

```bash
# Export (one-time)
make export-onnx IN=artifacts/runs/phase3-raw-balanced__arch=tcn_fm=raw_lr=0.0005_do=0.4/checkpoints/tcn_best.keras
# → artifacts/runs/.../tcn_best.onnx

# Parity check result
Max |Δ| vs Keras: 7.15e-07  (tolerance 1e-04) ✅

# Profile
make profile MODEL=artifacts/checkpoints/tcn_best.onnx
# → p95 = 0.23 ms, 4,890 fps
```

The ONNX model is **130× faster** than the Phase 4 target of 30 ms. At 4,890 fps, the server can handle ~163 simultaneous real-time sessions on a single CPU core.

---

## 8. Backend API

The Flask + Socket.IO server exposes:

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Server uptime, model SHA256, backend type, num_classes |
| `/metrics` | GET | Prometheus metrics (`?format=json` for JSON) |
| `/admin/reload` | POST | Hot-swap checkpoint without dropping WebSocket connections |
| `/rooms` | POST | Create a room |
| `/transcript` | GET | Conversation log for a room |
| `/feedback` | POST | User feedback submission |

**WebSocket events:**
```
client → server:  "frame"      {landmarks: [126 floats], t: unix_ms}
server → client:  "prediction" {label: str|null, confidence: float|null, ready: bool}
client → server:  "reset"
server → client:  "reset_ack"  {}
```

**Prediction smoother settings** (`PredictionSmoother`):
- EMA α = 0.6
- Confidence threshold = 0.75
- Repeat cooldown = 15 frames
- Idle TTL = 300 s (auto-evicts disconnected buffers)

---

## 9. Frontend

React + Next.js 15 App Router. Key routes:

| Route | Description |
|---|---|
| `/` | Landing page — create or join a room |
| `/practice` | Solo practice mode (no backend required after MediaPipe loads) |
| `/learn` | Sign dictionary with tips |
| `/r/[roomId]` | Live room — signer view or hearing view |
| `/r/[roomId]/join` | Role selection + name entry |

MediaPipe Hands runs **entirely in-browser** using the GPU delegate — no video leaves the device. Only the 126-float landmark vector per frame is sent to the server.

---

## 10. How to Run

```bash
# 1. Setup (one-time)
make setup              # pip install + label map + MediaPipe model download

# 2. Download training data (requires ~/.kaggle/kaggle.json)
python backend/scripts/download_datasets.py --dataset alphabet
python backend/scripts/download_datasets.py --dataset digits

# 3. Start backend (uses TCN ONNX by default)
export SIGNLEARN_SECRET_KEY=$(python -c 'import secrets; print(secrets.token_hex(32))')
make serve              # http://127.0.0.1:5001

# 4. Start frontend
make frontend           # http://localhost:3000

# 5. Health check
curl http://127.0.0.1:5001/health
```

**Environment variables:**

| Variable | Default | Description |
|---|---|---|
| `SIGNLEARN_SECRET_KEY` | *(required)* | Flask session secret |
| `SIGNLEARN_MODEL_PATH` | `artifacts/checkpoints/tcn_best.onnx` | Model checkpoint path |
| `SIGNLEARN_ADMIN_TOKEN` | *(unset)* | Enables `POST /admin/reload` hot-swap |

---

## 11. Next Steps

### Immediate (data)
- Record `k`, `v`, `r`, `u` with webcam at multiple angles (front, 30° left, 30° right) to expose z-depth — eliminates residual confusions
- Record 57 remaining vocabulary classes for full 93-class training

### Short-term (model)
- Train on 93 classes once data is collected (TCN architecture, same hyperparameters)
- Add `raw+velocity` features once dynamic word sequences (real video) are in the dataset — velocity carries meaningful motion signal for signs like `hello`, `please`, `thank_you`

### Long-term
- Add a bigram/trigram language model over the output stream to disambiguate `two`/`v` and `six`/`w` from context
- CTC decoder head for continuous phrase recognition (variable-length output without explicit segmentation)

---

## 12. Artifacts

| Artifact | Location |
|---|---|
| Winner checkpoint (ONNX) | `artifacts/checkpoints/tcn_best.onnx` |
| Winner checkpoint (Keras) | `artifacts/checkpoints/tcn_best.keras` |
| Full sweep report | `artifacts/reports/sweeps/phase3-raw-balanced.md` |
| Model comparison table | `artifacts/reports/model_comparison.md` |
| Inference profile | `artifacts/reports/inference_profile.md` |
| Data collection guide | `artifacts/reports/data_collection_recommendation.md` |
| Label map | `artifacts/label_map.json` |
| All sweep run histories | `artifacts/runs/phase3-raw-balanced__arch=*/reports/` |
