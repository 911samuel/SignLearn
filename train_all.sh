#!/usr/bin/env bash
# train_all.sh — train LSTM, BiLSTM, Transformer on 36-class dataset to completion.
# Launched with nohup; all output goes to /tmp/signlearn_train_all.log
# Usage: nohup bash train_all.sh > /tmp/signlearn_train_all.log 2>&1 &

set -e
cd "$(dirname "$0")"
export PYTHONPATH=.

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "=== SignLearn — Full training run (36 classes: digits + alphabet) ==="
log "Class weights capped at 5:1 to balance digit/letter imbalance"
log ""

ARCHS=(lstm bilstm transformer)
EPOCHS=100

for ARCH in "${ARCHS[@]}"; do
    RUN_NAME="${ARCH}-v2-36cls"
    log ">>> Starting: $ARCH  run-name=$RUN_NAME  epochs=$EPOCHS"
    python backend/scripts/train_model.py \
        --arch "$ARCH" \
        --run-name "$RUN_NAME" \
        --epochs "$EPOCHS"
    log "<<< Finished: $ARCH  (run=$RUN_NAME)"
    log ""
done

log "=== All architectures trained. Running evaluation ==="
python backend/scripts/evaluate_model.py \
    --runs lstm-v2-36cls bilstm-v2-36cls transformer-v2-36cls

log "=== Done. Results in artifacts/reports/model_comparison.md ==="
