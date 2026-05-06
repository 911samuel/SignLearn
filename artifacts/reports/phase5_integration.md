# Phase 5 Integration Report

**Date:** 2026-05-06  
**Branch:** `dev-web`  
**Status:** ✅ Complete

---

## Acceptance criteria

| # | Criterion | Result |
|---|-----------|--------|
| 5.1 | End-to-end pipeline runs on localhost (browser sign + speech → transcript) | ✅ Confirmed manually |
| 5.2 | End-to-end latency instrumented; p95 < 2000 ms | ✅ p95 = 127 ms |
| 5.3 | Latency optimisation | ⏭ Skipped — p95 well under target |
| 5.4 | Frontend resilience: webcam loss, server disconnect, reconnect banner | ✅ Verified manually |
| 5.5 | Backend resilience: server starts without model, /health truthful | ✅ Verified manually |
| 5.6 | Unified conversation log, hydration on mount, export button | ✅ Confirmed manually |
| 5.7 | Acceptance test suite passes | ✅ 5/5 pytest tests green |

---

## Latency

| Metric | Value | Target |
|--------|-------|--------|
| Mean WS round-trip | 40.7 ms | — |
| p50 | 35.9 ms | — |
| p95 | 127.1 ms | < 2000 ms |
| p99 | 127.1 ms | — |

Note: WS round-trip is a proxy for end-to-end latency. Add ~50–100 ms for
in-browser MediaPipe inference to get total capture-to-display time. Estimated
real-world p95 ≈ 200–230 ms — well under the 2 s proposal target.

Full report: `artifacts/reports/phase5_latency.json`

---

## Error-recovery checklist

| Scenario | Behaviour |
|----------|-----------|
| Backend stops mid-session | Red banner: "Server unavailable — predictions paused" |
| Backend reconnects | Banner clears, Socket.IO auto-reconnects (amber → green dot) |
| Model checkpoint missing on startup | Server stays alive; `/health` returns `model_loaded: false`; purple banner in frontend |
| Frame with wrong feature dimension | Backend emits `error` event with message; no crash |
| Camera permission denied | "Camera access denied" message in signer panel |
| Camera disconnected mid-session | "Camera disconnected" message; rAF loop pauses cleanly |
| Speech-to-text POST fails | `console.warn` logged; transcript continues locally |

---

## Automated test suite

```
pytest tests/test_phase5_integration.py -v

tests/test_phase5_integration.py::test_health_model_loaded              PASSED
tests/test_phase5_integration.py::test_ws_prediction_roundtrip          PASSED
tests/test_phase5_integration.py::test_speech_post_persists             PASSED
tests/test_phase5_integration.py::test_latency_report_passes            PASSED
tests/test_phase5_integration.py::test_model_not_loaded_emits_gracefully PASSED

5 passed in 0.57s
```

---

## Known limitations (Phase 6 scope)

- **Accuracy not validated** — the LSTM model is tested for structural correctness
  (inference runs, output shape is valid) but recognition accuracy against real ASL
  signs is deferred to Phase 6 usability sessions with DHH participants.
- **Mobile layout** — webcam panel at 640×480 requires horizontal scrolling on
  screens narrower than ~700 px. Responsive resizing is a Phase 6 enhancement.
- **Browser compatibility** — tested on Chrome/Safari (macOS). Firefox Web Speech
  API support is partial; speech-to-text may not work there.

---

## Handoff to Phase 6

Phase 5 is complete. The integrated prototype is ready for:

1. **Controlled environment testing** — measure accuracy/F1 under 3 lighting
   conditions and 2 backgrounds (Phase 6, Week 11).
2. **Usability sessions** — structured sessions with DHH individuals or
   interpreters (Phase 6, Week 12).
3. **Performance benchmarking** — FPS, memory usage, confusion matrix heatmap.
