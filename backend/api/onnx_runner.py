"""Thin onnxruntime wrapper exposing the same predict signature as Keras.

ONNX Runtime is ~10-20× faster than Keras on CPU for the sequence sizes we
serve (single-sample softmax over ~(30, 126..400) inputs). It also has no
Python-level GIL contention because the inference graph runs in C++, so the
existing `threading.Lock` around `model.predict` becomes unnecessary.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

_log = logging.getLogger(__name__)


class OnnxRunner:
    """Wrap an :class:`onnxruntime.InferenceSession` with a Keras-style API.

    Construction is lazy so importing this module never requires
    onnxruntime to be installed unless an .onnx checkpoint is actually used.
    """

    def __init__(self, path: Path) -> None:
        import onnxruntime as ort  # type: ignore[import-not-found]

        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 1  # tiny model, oversubscription hurts
        sess_opts.inter_op_num_threads = 1
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._session = ort.InferenceSession(str(path), sess_options=sess_opts)
        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name
        self._path = path
        _log.info(
            "OnnxRunner loaded %s (input=%s output=%s)",
            path.name, self._input_name, self._output_name,
        )

    def predict(self, x: np.ndarray, verbose: int = 0) -> np.ndarray:  # noqa: ARG002
        """Run inference on a single batch. Matches `tf.keras.Model.predict()` signature."""
        if x.dtype != np.float32:
            x = x.astype(np.float32, copy=False)
        out = self._session.run([self._output_name], {self._input_name: x})[0]
        return out

    def count_params(self) -> int:
        """Approximate weight count from ONNX initializers (best-effort)."""
        try:
            import onnx  # type: ignore[import-not-found]
            m = onnx.load(str(self._path))
            return sum(int(np.prod(t.dims)) for t in m.graph.initializer)
        except Exception:  # noqa: BLE001
            return 0

    @property
    def name(self) -> str:
        return self._path.name
