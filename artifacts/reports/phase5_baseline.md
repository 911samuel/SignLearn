# Phase 5 Baseline — Subtask 5.1

Date: 2026-05-06
Branch: `dev-web`

## Goal
Confirm the existing Phase 3+4 stack runs end-to-end on localhost before we touch any code.

## Environment
- Backend: `python backend/scripts/run_server.py` → http://127.0.0.1:5001
- Frontend: `cd frontend && npm run dev` → http://localhost:5173
- Model checkpoint: `artifacts/checkpoints/lstm_best.keras` (2.2 MB, 93 classes)

## Automated checks

| Check | Result |
|---|---|
| `GET /health` returns 200 | ✅ `{"status":"ok","feature_dim":126,"sequence_len":30,"model_loaded":false}` (model is lazy-loaded on first WS frame — expected) |
| `python tests/e2e_smoke.py` | ✅ passes — fixture frames, speech POST, transcript fetch all green |
| `python tests/profile_ws.py` baseline | _to run after browser smoke_ |

Note: `model_loaded:false` on a freshly started server is expected — `model_loader.py:get_model()` lazy-loads on first call. The e2e_smoke test confirms loading + inference path works.

## Manual browser flow (user)

Steps:
1. Start backend in one terminal, frontend in another.
2. Open http://localhost:5173, grant camera + mic permissions.
3. Sign at least 3 known classes (e.g. `a`, `b`, `hello`) — confirm predictions render in the signer panel.
4. Speak at least 1 short sentence — confirm transcript appears in the hearing panel.
5. `curl http://127.0.0.1:5001/transcript` — confirm both sources are persisted.

Capture:
- [ ] Screenshot 1: signer panel with a confident prediction
- [ ] Screenshot 2: hearing panel with a transcribed sentence
- [ ] Screenshot 3: `/transcript` JSON output showing both sources

Save screenshots under `artifacts/reports/phase5_baseline/`.

## Observed issues / notes

_(fill in during the browser session — anything weird about latency, layout, missing predictions, etc.)_

- 
- 

## Conclusion

_(once user confirms)_ Baseline established. Proceed to Subtask 5.2 (latency instrumentation).
