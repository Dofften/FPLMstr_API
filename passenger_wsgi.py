from a2wsgi import ASGIMiddleware
from app import app  # Import your FastAPI app.

application = ASGIMiddleware(app)