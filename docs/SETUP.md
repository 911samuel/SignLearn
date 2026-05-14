# SignLearn — Setup Guide

## Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.10 – 3.11 |
| Node.js | 18+ |
| pip | 23+ |
| Git LFS | any |

## 1. Clone and install dependencies

```bash
git clone <repo-url> SignLearn
cd SignLearn

# Pull Git LFS files (model checkpoints tracked via LFS)
git lfs pull

# Install Python dependencies
pip install -r requirements.txt
```

## 2. One-time data setup

```bash
# Build the label map from docs/vocabulary.md (required before any training)
python -m backend.data.label_map

# Download the MediaPipe hand landmarker model (~5 MB)
make model
```

## 3. Download training data (Kaggle)

You need a Kaggle account and an API key at `~/.kaggle/kaggle.json`.

```bash
# Download the ASL alphabet + digits datasets
python backend/scripts/download_datasets.py --dataset alphabet
python backend/scripts/download_datasets.py --dataset digits
```

## 4. Audit the dataset

```bash
make audit
# → artifacts/reports/dataset_audit.md
# → artifacts/reports/audit_signers.md
```

## 5. Train a model

```bash
# Quick smoke run (BiLSTM, raw features, 30 epochs)
make train ARCH=bilstm RUN_NAME=bilstm-smoke EPOCHS=30

# Production run (BiLSTM, engineered features, 80 epochs)
make train ARCH=bilstm FEATURE_MODE=engineered RUN_NAME=bilstm-eng-v1 EPOCHS=80

# See artifacts/runs/<run-name>/reports/ for metrics, confusion matrix, etc.
```

Available architectures: `lstm`, `bilstm`, `transformer`, `tcn`, `cnn_bilstm`, `conformer_lite`  
Available feature modes: `raw`, `raw+velocity`, `raw+velocity+angles`, `raw+velocity+acceleration`, `engineered`

## 6. Export to ONNX (for <30ms inference)

```bash
make export-onnx IN=artifacts/runs/bilstm-eng-v1/checkpoints/bilstm_best.keras
# → artifacts/runs/bilstm-eng-v1/bilstm_best.onnx

# Profile latency
make profile MODEL=artifacts/runs/bilstm-eng-v1/bilstm_best.onnx
# → artifacts/reports/inference_profile.md
```

## 7. Start the backend

```bash
# Required environment variables
export SIGNLEARN_SECRET_KEY=$(python -c 'import secrets; print(secrets.token_hex(32))')

# Start the server (http://127.0.0.1:5001)
make serve
```

To use the ONNX checkpoint instead of the default Keras one, set `SIGNLEARN_MODEL_PATH`:

```bash
export SIGNLEARN_MODEL_PATH=artifacts/runs/bilstm-eng-v1/bilstm_best.onnx
make serve
```

## 8. Start the frontend

```bash
make frontend
# → http://localhost:5173
```

## 9. Run tests

```bash
make test
# or a single file:
make test-file FILE=tests/test_augment.py
```

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `SIGNLEARN_SECRET_KEY` | *required in prod* | Flask session secret — generate with `secrets.token_hex(32)` |
| `SIGNLEARN_MODEL_PATH` | `artifacts/checkpoints/lstm_best.keras` | Path to the active checkpoint |
| `SIGNLEARN_ASYNC_MODE` | `threading` | Flask-SocketIO async mode (`threading` for tests) |
| `FLASK_DEBUG` | `0` | Set to `1` to enable debug mode (insecure default secret allowed) |
| `VITE_BACKEND_URL` | `http://127.0.0.1:5001` | Frontend → backend URL |
