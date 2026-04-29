# SignLearn Hardware Documentation

## Machine Specifications
- **CPU**: Apple M2 Pro (12-core)
- **RAM**: 16 GB LPDDR5
- **GPU**: Integrated 19-core GPU

## GPU Availability Result
- **TensorFlow Detection**: Detected via `tensorflow-metal` (PluggableDevice).
- **Status**: Available for hardware acceleration.

## Implementation Decision
- **Training Strategy**: **CPU-Oriented with GPU Acceleration.**
- **Rationale**: 
  - MediaPipe Hands runs optimally on CPU for real-time inference.
  - The LSTM model architecture (landmark-based) is small enough that CPU training is viable, though GPU will be used to speed up hyperparameter tuning.
  - No massive CNN layers are present, reducing the strict dependency on high-end VRAM.

## Impact on Strategy
- **Data Pipeline**: Landmark extraction must be decoupled from training to avoid CPU bottlenecks.
- **Real-time Target**: 30 FPS on CPU is the primary benchmark for production readiness.
