"""Single source of truth for shared pipeline constants.

All data-pipeline modules (extract, normalize, dataset, validate) and the
model-config module import from here so the numbers are never repeated.

Do NOT import from other backend modules here — this file must be importable
without triggering any side-effects or circular dependencies.
"""

#: Frames captured / expected per sequence
SEQUENCE_LEN: int = 30

#: MediaPipe landmarks per hand
N_LANDMARKS: int = 21

#: Coordinates per landmark (x, y, z)
COORDS: int = 3

#: Feature dimension for one hand  (21 × 3 = 63)
HAND_DIM: int = N_LANDMARKS * COORDS

#: Feature dimension for both hands (63 × 2 = 126) — the LSTM input width
FEATURE_DIM: int = HAND_DIM * 2

# Alias kept for modules that historically used TWO_HAND_DIM
TWO_HAND_DIM: int = FEATURE_DIM
