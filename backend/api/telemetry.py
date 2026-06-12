"""In-process telemetry collector for SignLearn.

Tracks prediction counts, latency histograms, no-hand-frame rate, and
per-class rolling confidence means. Thread-safe via a single RLock.

Exposed via ``GET /metrics`` in Prometheus text format (no external deps —
just ``text/plain; version=0.0.4``), and as structured JSON via
``GET /metrics?format=json`` for programmatic consumers.

Usage::

    from backend.api.telemetry import METRICS
    METRICS.record_prediction(label="hello", confidence=0.88, latency_ms=12.4)
    METRICS.record_no_hand_frame()
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from typing import Dict, List

# Latency histogram bucket upper bounds (ms) — covers single-sample CPU inference.
_LATENCY_BUCKETS = (5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500, 1000)


class _Histogram:
    """Simple fixed-bucket histogram (not thread-safe — caller holds lock)."""

    def __init__(self, buckets: tuple) -> None:
        self._buckets = buckets
        self._counts = [0] * len(buckets)
        self._inf = 0
        self._sum = 0.0
        self._total = 0

    def observe(self, value: float) -> None:
        self._sum += value
        self._total += 1
        for i, bound in enumerate(self._buckets):
            if value <= bound:
                self._counts[i] += 1
                return
        self._inf += 1

    def prometheus_lines(self, name: str, labels: str = "") -> List[str]:
        lines = []
        cumulative = 0
        for i, bound in enumerate(self._buckets):
            cumulative += self._counts[i]
            if labels:
                bucket_label = '{' + labels + ',le="' + str(bound) + '"}'
            else:
                bucket_label = '{le="' + str(bound) + '"}'
            lines.append(f"{name}_bucket{bucket_label} {cumulative}")
        cumulative += self._inf
        if labels:
            inf_label = '{' + labels + ',le="+Inf"}'
        else:
            inf_label = '{le="+Inf"}'
        lines.append(f"{name}_bucket{inf_label} {cumulative}")
        label_str = "{" + labels + "}" if labels else ""
        lines.append(f"{name}_sum{label_str} {self._sum:.4f}")
        lines.append(f"{name}_count{label_str} {self._total}")
        return lines

    def to_dict(self) -> dict:
        return {
            "sum_ms": round(self._sum, 3),
            "count": self._total,
            "mean_ms": round(self._sum / self._total, 3) if self._total else 0.0,
            "buckets": {str(b): c for b, c in zip(self._buckets, self._counts)},
            "inf": self._inf,
        }


class MetricsCollector:
    """Thread-safe in-process metrics store."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._start_time = time.time()
        # Counters
        self._prediction_count = 0
        self._no_hand_frames = 0
        self._total_frames = 0
        self._suppressed_count = 0  # smoothed out / below threshold
        # Histograms
        self._latency = _Histogram(_LATENCY_BUCKETS)
        # Per-class rolling confidence (EMA alpha=0.05)
        self._class_conf: Dict[str, float] = defaultdict(float)
        self._class_count: Dict[str, int] = defaultdict(int)

    # -- write path --

    def record_prediction(
        self,
        label: str | None,
        confidence: float | None,
        latency_ms: float,
    ) -> None:
        with self._lock:
            self._total_frames += 1
            self._latency.observe(latency_ms)
            if label is None:
                self._suppressed_count += 1
            else:
                self._prediction_count += 1
                if confidence is not None:
                    alpha = 0.05
                    prev = self._class_conf.get(label, confidence)
                    self._class_conf[label] = alpha * confidence + (1.0 - alpha) * prev
                    self._class_count[label] += 1

    def record_no_hand_frame(self) -> None:
        with self._lock:
            self._no_hand_frames += 1
            self._total_frames += 1

    # -- read path --

    def to_prometheus(self) -> str:
        with self._lock:
            uptime = time.time() - self._start_time
            lines = [
                "# HELP signlearn_uptime_seconds Server uptime in seconds",
                "# TYPE signlearn_uptime_seconds gauge",
                f"signlearn_uptime_seconds {uptime:.1f}",
                "",
                "# HELP signlearn_predictions_total Total emitted predictions (label not suppressed)",
                "# TYPE signlearn_predictions_total counter",
                f"signlearn_predictions_total {self._prediction_count}",
                "",
                "# HELP signlearn_frames_total Total landmark frames received",
                "# TYPE signlearn_frames_total counter",
                f"signlearn_frames_total {self._total_frames}",
                "",
                "# HELP signlearn_no_hand_frames_total Frames with no detected hand",
                "# TYPE signlearn_no_hand_frames_total counter",
                f"signlearn_no_hand_frames_total {self._no_hand_frames}",
                "",
                "# HELP signlearn_suppressed_total Frames suppressed by smoother",
                "# TYPE signlearn_suppressed_total counter",
                f"signlearn_suppressed_total {self._suppressed_count}",
                "",
                "# HELP signlearn_inference_latency_ms Inference latency histogram (ms)",
                "# TYPE signlearn_inference_latency_ms histogram",
                *self._latency.prometheus_lines("signlearn_inference_latency_ms"),
                "",
            ]
            # Per-class rolling confidence
            if self._class_conf:
                lines += [
                    "# HELP signlearn_class_confidence_rolling Per-class rolling mean confidence (EMA)",
                    "# TYPE signlearn_class_confidence_rolling gauge",
                ]
                for cls, conf in sorted(self._class_conf.items()):
                    lines.append(f'signlearn_class_confidence_rolling{{class="{cls}"}} {conf:.4f}')
                lines.append("")
            return "\n".join(lines)

    def to_dict(self) -> dict:
        with self._lock:
            no_hand_rate = (
                self._no_hand_frames / self._total_frames
                if self._total_frames else 0.0
            )
            return {
                "uptime_seconds": round(time.time() - self._start_time, 1),
                "predictions_total": self._prediction_count,
                "frames_total": self._total_frames,
                "no_hand_frames_total": self._no_hand_frames,
                "no_hand_rate": round(no_hand_rate, 4),
                "suppressed_total": self._suppressed_count,
                "inference_latency_ms": self._latency.to_dict(),
                "class_confidence_rolling": {
                    cls: round(conf, 4) for cls, conf in sorted(self._class_conf.items())
                },
            }


# Module-level singleton — import and call directly.
METRICS = MetricsCollector()
