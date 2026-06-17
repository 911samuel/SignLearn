FROM python:3.11-slim

WORKDIR /app

# libgomp1 is required by onnxruntime; build-essential covers wheels that need
# native compilation on linux/arm64 base images.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgomp1 \
 && rm -rf /var/lib/apt/lists/*

COPY requirements-prod.txt ./
RUN pip install --no-cache-dir -r requirements-prod.txt

# Copy only what the server needs at runtime.  Training scripts, datasets,
# notebooks, and the frontend stay out of the image.
COPY backend/ ./backend/
COPY artifacts/checkpoints/tcn_best.onnx ./artifacts/checkpoints/tcn_best.onnx
COPY artifacts/checkpoints/tcn_best.classes.json ./artifacts/checkpoints/tcn_best.classes.json
COPY artifacts/checkpoints/tcn_word_best.onnx ./artifacts/checkpoints/tcn_word_best.onnx
COPY artifacts/label_map.json ./artifacts/label_map.json
COPY docs/vocabulary.md ./docs/vocabulary.md
COPY configs/asl_citizen_demo_words_curated.txt ./configs/asl_citizen_demo_words_curated.txt
COPY configs/word_curated_v6_27cls.txt ./configs/word_curated_v6_27cls.txt

ENV PYTHONUNBUFFERED=1 \
    SIGNLEARN_ASYNC_MODE=gevent \
    SIGNLEARN_MODEL_PATH=/app/artifacts/checkpoints/tcn_best.onnx \
    PORT=8000

EXPOSE 8000

# Single gevent worker — Flask-SocketIO needs sticky sessions, and one gevent
# worker handles thousands of concurrent WebSocket clients via its event loop.
# (Gunicorn 26 dropped the eventlet worker; gevent is the supported equivalent.)
# --timeout 120 covers slow ONNX cold loads on small instances.
CMD ["gunicorn", \
     "--worker-class", "geventwebsocket.gunicorn.workers.GeventWebSocketWorker", \
     "--workers", "1", \
     "--bind", "0.0.0.0:8000", \
     "--timeout", "120", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "backend.api.wsgi:application"]
