.PHONY: setup model label-map audit train serve test frontend clean-processed help

# ──────────────────────────────────────────────────────────────────────────────
# Variables (override on the command line: make train ARCH=tcn)
# ──────────────────────────────────────────────────────────────────────────────
ARCH        ?= bilstm
FEATURE_MODE?= raw
RUN_NAME    ?= $(ARCH)-v1
EPOCHS      ?= 60
SWEEP_CFG   ?= configs/sweeps/phase3.yaml

PYTHON      := python
PYTEST      := SIGNLEARN_ASYNC_MODE=threading SIGNLEARN_SECRET_KEY=test-secret pytest

# ──────────────────────────────────────────────────────────────────────────────
# Setup
# ──────────────────────────────────────────────────────────────────────────────

## Install Python dependencies
setup: requirements.txt
	pip install -r requirements.txt
	$(MAKE) label-map model

## Download the MediaPipe hand landmarker (one-time, ~5 MB)
model:
	@mkdir -p models
	curl -L https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task \
	     -o models/hand_landmarker.task
	@echo "Saved → models/hand_landmarker.task"

## Build artifacts/label_map.json from docs/vocabulary.md
label-map:
	$(PYTHON) -m backend.data.label_map
	@echo "Label map written → artifacts/label_map.json"

# ──────────────────────────────────────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────────────────────────────────────

## Audit dataset quality (writes artifacts/reports/dataset_audit.md + audit_signers.md)
audit:
	$(PYTHON) backend/scripts/audit_dataset.py

TARGET ?= 600
## Augment minority classes to TARGET samples: make augment TARGET=600
augment:
	$(PYTHON) backend/scripts/augment_minority.py --target-count $(TARGET)

## Record a new landmark sequence from webcam (usage: make record LABEL=hello)
record:
	$(PYTHON) backend/scripts/record_vocabulary.py --words $(LABEL)

# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

## Train a single run: make train ARCH=bilstm FEATURE_MODE=engineered RUN_NAME=bilstm-eng-v1
train:
	$(PYTHON) backend/scripts/train_model.py \
	    --arch $(ARCH) \
	    --feature-mode $(FEATURE_MODE) \
	    --run-name $(RUN_NAME) \
	    --epochs $(EPOCHS)

## Run a hyperparameter sweep: make sweep SWEEP_CFG=configs/sweeps/phase3_smoke.yaml
sweep:
	$(PYTHON) backend/scripts/sweep.py --config $(SWEEP_CFG)

## Evaluate + compare all runs in artifacts/runs/
compare:
	$(PYTHON) backend/scripts/evaluate_model.py

## Evaluate a single run: make eval RUN=bilstm-v2-36cls
RUN ?= bilstm-v2-36cls
eval:
	$(PYTHON) backend/scripts/evaluate_model.py --runs $(RUN)

# ──────────────────────────────────────────────────────────────────────────────
# ONNX export and profiling
# ──────────────────────────────────────────────────────────────────────────────

## Export best keras checkpoint to ONNX (usage: make export-onnx IN=path/to/best.keras)
export-onnx:
	$(PYTHON) backend/scripts/export_onnx.py --in $(IN)

## Profile inference latency on CPU (usage: make profile MODEL=path/to/model.onnx)
profile:
	$(PYTHON) backend/scripts/profile_inference.py --model $(MODEL) --backend auto

# ──────────────────────────────────────────────────────────────────────────────
# Serving
# ──────────────────────────────────────────────────────────────────────────────

## Start the Flask/SocketIO backend with default model (http://127.0.0.1:5001)
serve:
	$(PYTHON) backend/scripts/run_server.py

## Start the backend serving a specific ONNX checkpoint (usage: make serve-onnx MODEL=path/to/model.onnx)
MODEL ?= artifacts/checkpoints/tcn_best.onnx
serve-onnx:
	SIGNLEARN_MODEL_PATH=$(MODEL) $(PYTHON) backend/scripts/run_server.py

## Start the Next.js frontend dev server (http://localhost:5173)
frontend:
	cd frontend && npm install && npm run dev

# ──────────────────────────────────────────────────────────────────────────────
# Testing
# ──────────────────────────────────────────────────────────────────────────────

## Run the full test suite
test:
	$(PYTEST) -v

## Run a single test file: make test-file FILE=tests/test_augment.py
test-file:
	$(PYTEST) $(FILE) -v

# ──────────────────────────────────────────────────────────────────────────────
# Maintenance
# ──────────────────────────────────────────────────────────────────────────────

## Delete all processed .npy sequences (keeps raw data and artifacts)
clean-processed:
	find data/processed -name "*.npy" -delete
	@echo "Removed all .npy files from data/processed/"

## Show this help message
help:
	@awk '/^## /{desc=substr($$0,4); next} /^[a-zA-Z_-]+:/{print "\033[36m" $$1 "\033[0m", desc; desc=""}' $(MAKEFILE_LIST) | column -t -s ' '

.DEFAULT_GOAL := help
