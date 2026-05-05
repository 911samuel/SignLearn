.PHONY: phase1 model test clean-processed

# Download the MediaPipe hand landmarker model (one-time setup)
model:
	curl -L https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task \
	     -o models/hand_landmarker.task

# Run the full Phase 1 data pipeline
phase1: artifacts/label_map.json
	python -m backend.data.extract --raw data/raw --out data/processed --workers 4
	python -m backend.data.validate --processed data/processed

# Build the label map (prerequisite for phase1)
artifacts/label_map.json:
	python -m backend.data.label_map

# Run all tests
test:
	pytest

# Delete processed sequences (keeps raw data and artifacts)
clean-processed:
	find data/processed -name "*.npy" -delete
	@echo "Removed all .npy files from data/processed/"
