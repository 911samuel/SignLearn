# Going Live: Deployment Plan

**Goal:** A single shareable URL (e.g. `https://signlearn.vercel.app`) where a hearing user and a signer can join a room and converse. Target: 1–2 weeks to demo.

**Architecture (decided):**

```
Browser (webcam + MediaPipe)
        │  HTTPS + WSS
        ▼
[ Vercel — Next.js frontend ]   ───►   [ Render — Flask + SocketIO + ONNX ]
   signlearn.vercel.app                signlearn-api.onrender.com
```

Frontend on Vercel (free), backend on Render (Starter $7/mo so it never sleeps mid-demo). Total ongoing cost: **$7/month**.

---

## Why not all-Vercel

Vercel Functions are short-lived and don't keep per-connection state. Your backend has:
- A `FrameBuffer` per `request.sid` that accumulates 30 frames before predicting.
- `PredictionSmoother` EMA state per signer.
- A loaded ONNX model held in a module-level `ModelHolder`.

That model needs a long-running container. Render gives you exactly that with a 5-line config.

---

## Status: what's already done (Steps 1 + 2)

All code-level changes for deploy are committed in this branch. The production Docker image builds, boots in ~5 seconds, and loads the ONNX model with all 36 class names registered. Verified locally with `docker run`.

Files that landed:

| File | Why |
|---|---|
| [Dockerfile](../Dockerfile) | Python 3.11-slim, slim deps, gunicorn + gevent-websocket worker, eager model load |
| [.dockerignore](../.dockerignore) | Keeps build context to ~30MB (excludes frontend, tests, training data) |
| [render.yaml](../render.yaml) | Render Blueprint — service definition, plan, env vars |
| [requirements-prod.txt](../requirements-prod.txt) | Slim runtime deps (no TF, pytest, Kaggle, matplotlib) |
| [backend/api/wsgi.py](../backend/api/wsgi.py) | Gunicorn entry point; eager-loads the model at worker boot |
| [backend/api/model_loader.py](../backend/api/model_loader.py) | Reads class names from a JSON sidecar; lazy-imports the TF-backed fallback only when sidecar is missing |
| [backend/api/config.py](../backend/api/config.py) | CORS origins now read from `SIGNLEARN_CORS_ORIGINS` env var |
| [artifacts/checkpoints/tcn_best.onnx](../artifacts/checkpoints/tcn_best.onnx) | Pinned at the default path so the Docker image is self-contained |
| [artifacts/checkpoints/tcn_best.classes.json](../artifacts/checkpoints/tcn_best.classes.json) | Class-names sidecar (36 classes) shipped next to the ONNX |
| [.gitignore](../.gitignore) | Exceptions so the ONNX + sidecar + label_map ship with the repo |

### Real bugs caught by the Docker build (would have crashed on Render)

1. **gunicorn 26 removed the eventlet worker.** The original plan used `--worker-class eventlet`. That class is gone in gunicorn ≥23. Switched to `gevent` + `gevent-websocket` (still actively maintained, first-class Flask-SocketIO support).
2. **The reachable import chain pulled in TensorFlow** via `compact_class_names → backend.data.dataset → import tensorflow`. Caught at container boot with `ModuleNotFoundError: No module named 'tensorflow'`. Refactored `model_loader.py` to read a `<model>.classes.json` sidecar and only fall back to the TF-backed function in dev (when the sidecar is absent).
3. **Class names would have been empty in production.** Even without the TF problem, `compact_class_names` scans `data/processed/train/` which isn't in the container. The sidecar fixes both issues with one stone.

### Why these matter

Without the Docker verification, the first sign that anything was wrong would have been Render saying "deployment failed" with a stack trace 10 minutes into a build — much harder to debug remotely than locally. Worth the half hour.

---

## What's left

Two dashboards and one smoke test. ~30 minutes total once an account exists on each.

### Step 1 — Deploy backend to Render

1. Push this branch (or merge to `main` — Render's `autoDeploy: true` listens for `main`).
2. On render.com: **New → Blueprint → connect GitHub → pick this repo**. Render reads [render.yaml](../render.yaml) and provisions the service.
3. In the service settings, fill the two env vars that [render.yaml](../render.yaml) marks `sync: false`:
   - `SIGNLEARN_ADMIN_TOKEN` — generate locally with `python -c "import secrets; print(secrets.token_hex(32))"` and paste.
   - `SIGNLEARN_CORS_ORIGINS` — leave empty for now; you'll fill it after Step 2.
4. Wait for the first build. ~3–5 min for `pip install`, ~30s for image push. The container should log:
   ```
   [INFO] Listening at: http://0.0.0.0:8000
   [INFO] Using worker: geventwebsocket.gunicorn.workers.GeventWebSocketWorker
   [INFO] Booting worker with pid: 7
   ```
5. Confirm the service is healthy: `curl https://signlearn-api.onrender.com/health`. Expect:
   ```json
   {
     "backend": "onnx",
     "model_loaded": true,
     "num_classes": 36,
     "model_sha256": "8947bdb6...",
     "status": "ok"
   }
   ```

### Step 2 — Deploy frontend to Vercel

The frontend is a standard Next.js 15 App Router project — Vercel auto-detects.

1. On vercel.com: **Add New → Project → import this repo**.
2. **Root directory:** `frontend/` (important — repo root has Python).
3. **Environment variable:** `NEXT_PUBLIC_BACKEND_URL = https://signlearn-api.onrender.com` (whatever URL Render gave you).
4. Deploy. You'll get `signlearn-<hash>.vercel.app`. Optionally promote a clean subdomain in **Settings → Domains** → `signlearn.vercel.app` is free.

### Step 3 — Close the CORS loop

Go back to the Render dashboard and set:

```
SIGNLEARN_CORS_ORIGINS = https://signlearn.vercel.app
```

(Add a comma-separated second entry like `https://signlearn-git-feature-model-training-yourname.vercel.app` if you want preview deploys to work too.)

Click **Manual Deploy → Clear cache & deploy** to apply the new env var.

### Step 4 — Smoke test

From a phone on cellular (not your home wifi — proves it works for your supervisor):

1. Visit `https://signlearn.vercel.app`. Page loads, no console errors.
2. Click "Start signing" → grant camera permission. Only works over HTTPS; Vercel is HTTPS by default.
3. Browser DevTools → Network → WS → confirm a `wss://signlearn-api.onrender.com/socket.io/...` connection is open and frames are flowing.
4. Sign a letter (e.g. `A`) and check a prediction appears within ~1 second.
5. Open a second browser/phone, join the same room code, confirm the hearing-user panel receives the transcript.

If something fails, the fastest debugging path:
- `https://signlearn-api.onrender.com/health` → confirms backend is up and which model SHA is loaded.
- `https://signlearn-api.onrender.com/metrics?format=json` → confirms predictions are being made.
- Render dashboard → Logs → live tail.

---

## Day-of-demo checklist

- [ ] `/health` returns 200 with `"model_loaded": true` and the expected SHA.
- [ ] Open the live URL on your phone *and* a clean browser profile (no localStorage state).
- [ ] Confirm the Render plan is **Starter or higher** (free tier sleeps after 15 min and the first request takes ~50 seconds to cold-start — disastrous mid-presentation).
- [ ] Pre-warm by hitting the URL 5 minutes before. WebSocket handshake should be <500ms.
- [ ] Have a backup: keep `make serve` + `make frontend` running locally with ngrok ready as fallback if Render has an outage.

---

## What this plan does *not* cover (acceptable for a final-year demo)

- **Rate limiting / abuse protection.** Anyone with the URL can use it. Fine for a supervised demo, not a public launch.
- **Auth.** No login. Room codes are the only access control.
- **Horizontal scaling.** A single gevent worker handles ~100 concurrent signers comfortably. If you expect more (you won't for a demo), add Redis as the SocketIO message queue and scale workers.
- **Persistent transcripts across restarts.** SQLite at `artifacts/signlearn.sqlite` lives inside the container — wiped on each deploy. For the demo this is fine; for production, move to Render's managed Postgres ($7/mo) and point `db_path` at it via a `DATABASE_URL` env var.
- **Custom domain.** `signlearn.vercel.app` is free and works. Buy a real domain only if your supervisor specifically wants one.

---

## Rough timeline (1–2 weeks)

| Days | Work | Status |
|---|---|---|
| 1 | Pre-flight code fixes (CORS env-driven, slim requirements, pin ONNX, class-names sidecar) | done |
| 2 | Dockerfile + `wsgi.py` + `render.yaml`; local Docker build verified | done |
| 3 | Push to GitHub; Render Blueprint deploy; debug until `/health` is green | next |
| 4 | Deploy frontend to Vercel; wire `NEXT_PUBLIC_BACKEND_URL`; close CORS loop | next |
| 5 | Smoke test from phone on cellular | next |
| 6–7 | Buffer for bugs found in cellular testing (latency, WSS upgrade issues) | |
| 8+ | Polish: custom domain, basic abuse rate limit, supervisor walkthrough rehearsal | |
