"""Subtask 5: render landmark sequences as skeleton overlays for visual debugging."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from backend.data.normalize import (
    HAND_DIM,
    N_LANDMARKS,
    TWO_HAND_DIM,
    normalize_sequence,
)

# Reused from scripts/extract_landmarks.py
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),           # index
    (5, 9), (9, 10), (10, 11), (11, 12),      # middle
    (9, 13), (13, 14), (14, 15), (15, 16),    # ring
    (13, 17), (17, 18), (18, 19), (19, 20),   # pinky
    (0, 17),                                   # palm base
]

# Colours: left hand green, right hand blue
_COLOURS = {
    "left":  (0, 220, 0),
    "right": (0, 120, 255),
}
_JOINT_RADIUS  = 5
_BONE_THICKNESS = 2


def _hand_to_pixels(
    hand: np.ndarray,
    canvas_size: int,
    margin: float = 0.1,
) -> list[tuple[int, int]]:
    """Convert (21, 3) hand landmarks to pixel coords on a square canvas.

    Landmarks after normalization are roughly in [-1.5, 1.5].  We remap the
    xy range to [margin, 1-margin] of the canvas so the skeleton sits centred
    with some padding.
    """
    # Use x (col) and y (row) only; z is ignored for 2-D drawing
    xy = hand[:, :2]  # (21, 2)

    # Skip all-zero hands
    if not np.any(xy):
        return []

    # Re-scale from arbitrary float range to [margin, 1-margin]
    min_xy = xy.min(axis=0)
    max_xy = xy.max(axis=0)
    span = max_xy - min_xy
    span[span < 1e-9] = 1.0   # avoid div-by-zero for degenerate hands

    normed = (xy - min_xy) / span                        # [0, 1]
    padded = normed * (1 - 2 * margin) + margin          # [margin, 1-margin]
    pixels = (padded * canvas_size).astype(int)

    return [(int(p[0]), int(p[1])) for p in pixels]


def draw_skeleton(
    frame: np.ndarray,
    canvas_size: int = 480,
    normalized: bool = True,
) -> np.ndarray:
    """Draw both hand skeletons from a (126,) landmark frame onto a blank canvas.

    Args:
        frame:        (126,) float32 — [left(63) | right(63)]
        canvas_size:  side length in pixels of the square output image
        normalized:   if True the frame is already wrist-centred/scaled;
                      if False normalization is applied before drawing

    Returns:
        (canvas_size, canvas_size, 3) uint8 BGR image
    """
    assert frame.shape == (TWO_HAND_DIM,), f"Expected ({TWO_HAND_DIM},), got {frame.shape}"

    if not normalized:
        from backend.data.normalize import normalize_frame
        frame = normalize_frame(frame)

    canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)

    halves = [
        ("left",  frame[:HAND_DIM].reshape(N_LANDMARKS, 3)),
        ("right", frame[HAND_DIM:].reshape(N_LANDMARKS, 3)),
    ]

    for side, hand in halves:
        pts = _hand_to_pixels(hand, canvas_size)
        if not pts:
            continue
        colour = _COLOURS[side]
        for a, b in HAND_CONNECTIONS:
            cv2.line(canvas, pts[a], pts[b], colour, _BONE_THICKNESS)
        for pt in pts:
            cv2.circle(canvas, pt, _JOINT_RADIUS, colour, -1)

    return canvas


def render_sequence(
    seq: np.ndarray,
    normalized: bool = True,
    canvas_size: int = 480,
) -> list[np.ndarray]:
    """Return a list of BGR images (one per frame) for a (T, 126) sequence."""
    assert seq.ndim == 2 and seq.shape[1] == TWO_HAND_DIM
    return [draw_skeleton(seq[t], canvas_size=canvas_size, normalized=normalized)
            for t in range(seq.shape[0])]


def play_window(seq: np.ndarray, title: str = "SignLearn Skeleton", fps: int = 10) -> None:
    """Display the skeleton sequence in an OpenCV window (press q to quit)."""
    frames = render_sequence(seq)
    delay  = max(1, 1000 // fps)
    while True:
        for img in frames:
            cv2.imshow(title, img)
            key = cv2.waitKey(delay)
            if key & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                return
    cv2.destroyAllWindows()


def save_gif(seq: np.ndarray, out_path: Path, fps: int = 10) -> None:
    """Write the skeleton sequence to an animated GIF using Pillow."""
    from PIL import Image

    frames = render_sequence(seq)
    duration_ms = 1000 // fps

    pil_frames = []
    for bgr in frames:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil_frames.append(Image.fromarray(rgb))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pil_frames[0].save(
        str(out_path),
        save_all=True,
        append_images=pil_frames[1:],
        loop=0,
        duration=duration_ms,
    )
    print(f"Saved GIF → {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize a (30,126) .npy landmark sequence as a skeleton animation."
    )
    parser.add_argument("--npy",        required=True,        help="Path to .npy file")
    parser.add_argument("--out",        default=None,         help="Save as GIF (omit for live window)")
    parser.add_argument("--normalize",  action="store_true",  help="Apply normalization before drawing")
    parser.add_argument("--fps",        type=int, default=10, help="Frames per second")
    parser.add_argument("--size",       type=int, default=480, help="Canvas size in pixels")
    args = parser.parse_args()

    npy_path = Path(args.npy)
    if not npy_path.exists():
        raise FileNotFoundError(npy_path)

    seq = np.load(str(npy_path)).astype(np.float32)
    if args.normalize:
        seq = normalize_sequence(seq)

    print(f"Loaded {npy_path.name}: shape={seq.shape}, "
          f"range=[{seq.min():.3f}, {seq.max():.3f}]")

    if args.out:
        save_gif(seq, Path(args.out), fps=args.fps)
    else:
        play_window(seq, title=npy_path.stem, fps=args.fps)
