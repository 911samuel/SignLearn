# SignLearn — Real-Time Sign Language Translator

> Two people, one room: a deaf or hard-of-hearing signer signs to the camera, a hearing user speaks into the mic, and both see live captions of each other.

**Live demo:** [sign-learn-one.vercel.app](https://sign-learn-one.vercel.app) · **Final-year project**, BSc Computer & Software Engineering, University of Rwanda

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?logo=tensorflow&logoColor=white)
![ONNX](https://img.shields.io/badge/ONNX-Runtime-005CED?logo=onnx&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-Socket.IO-000000?logo=flask&logoColor=white)
![Next.js](https://img.shields.io/badge/Next.js-15-000000?logo=nextdotjs&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Render-2496ED?logo=docker&logoColor=white)

---

## Results at a glance

| Metric | Result | Target |
|---|---:|---:|
| Test accuracy (36 classes: a–z, 0–9) | **97.84%** | ≥ 95% |
| Macro F1 | **95.45%** | ≥ 92% |
| Inference latency (ONNX, CPU, p95) | **0.23 ms** | < 30 ms |
| Throughput (single CPU core) | **~4,890 fps** | — |
| Model size | **126K parameters** | — |

Accuracy went from a **73.4% LSTM baseline → 84.1% BiLSTM → 97.84% TCN** after a systematic 16-run sweep (4 architectures × 2 feature sets × 2 learning rates).

---

## How it works

```
Browser (Next.js 15)
  └─ MediaPipe Hands, in-browser (GPU)  → 126 floats/frame (2 hands × 21 landmarks × xyz)
       └─ WebSocket "frame" event
            └─ Flask + Socket.IO backend
                 └─ 30-frame sliding window → normalisation (wrist-centred, unit-scaled)
                      └─ TCN model (ONNX Runtime, CPU)
                           └─ Prediction smoother (EMA + 0.75 confidence gate)
                                └─ WebSocket "prediction" event → live captions
```

**Key design decisions**

- **Privacy by design.** Hand tracking runs in the browser, so no video ever leaves the device. Only 126 numbers per frame reach the server.
- **TCN over LSTM.** Dilated 1D convolutions cover the full 30-frame window, train in parallel, and beat the best BiLSTM with 3.4× fewer parameters.
- **ONNX for serving.** The production image needs no TensorFlow. Keras-to-ONNX parity was verified (max |Δ| = 7.15e-07).
- **Stable captions.** An EMA smoother with a confidence threshold and repeat cooldown stops flickering predictions.

Full diagrams (system, sequence, training pipeline, frontend, model): [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)

---

## Features

- **Live rooms** (`/r/[roomId]`): signer and hearing views with sign→text captions and speech→text (Web Speech API)
- **Practice mode** (`/practice`): solo signing with a real-time confidence meter and landmark overlay
- **Sign dictionary** (`/learn`) with tips
- **Ops endpoints**: `/health` (model SHA-256, status), `/metrics` (Prometheus), `/admin/reload` (hot-swap model without dropping WebSocket connections)
- Conversation transcripts and feedback stored in SQLite

---

## Tech stack

| Layer | Technology |
|---|---|
| ML | Python, TensorFlow/Keras, MediaPipe, ONNX Runtime, TensorBoard |
| Backend | Flask, Flask-SocketIO, gunicorn + gevent-websocket, SQLite |
| Frontend | Next.js 15 (App Router), React, TypeScript, Socket.IO client |
| Quality | pytest, GitHub Actions CI |
| Deployment | Docker, Render (API), Vercel (frontend) |

---

## Run locally

```bash
make setup                 # install deps, label map, MediaPipe model
export SIGNLEARN_SECRET_KEY=$(python -c 'import secrets; print(secrets.token_hex(32))')
make serve                 # backend  → http://127.0.0.1:5001
make frontend              # frontend → http://localhost:3000
curl http://127.0.0.1:5001/health
pytest                     # run the test suite
```

Training data (Kaggle ASL Alphabet + Digits) and the full training/sweep workflow: [`docs/SETUP.md`](docs/SETUP.md), [`docs/RUNBOOK.md`](docs/RUNBOOK.md)

---

## Engineering notes

- **The Docker build caught three production bugs before deploy:** gunicorn ≥ 23 dropped the eventlet worker (moved to gevent), a hidden import chain pulled TensorFlow into the slim image, and class names would have been empty in the container (fixed with a JSON sidecar next to the ONNX model).
- **Known limitations, stated honestly:** `two`↔`v` and `six`↔`w` share the same ASL handshape, so telling them apart needs sentence context (planned: n-gram language model). The current model covers 36 static classes. The 93-class vocabulary with dynamic word signs needs recorded video data.

Full report with every sweep run and per-class metrics: [`docs/REPORT.md`](docs/REPORT.md) · Deployment: [`docs/deployment.md`](docs/deployment.md)

---

## Author

**Samuel Abayizera** · [GitHub](https://github.com/911samuel) · [LinkedIn](https://www.linkedin.com/in/abayizera-samuel-184845345)
