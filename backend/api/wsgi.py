"""Production WSGI entry point used by gunicorn.

`gunicorn --worker-class eventlet backend.api.wsgi:application` will find
`application` here and serve both HTTP and Socket.IO traffic through it.
"""

from backend.api.app import create_app
from backend.api.model_loader import _holder

app, socketio = create_app()
application = app

# Eager-load the ONNX model at worker boot so the first signer doesn't pay
# the cold-load cost, and any load failure surfaces in logs immediately
# instead of at first frame.
_holder.load()
