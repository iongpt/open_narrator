"""
OpenNarrator - Self-hosted audio translation service.

Main FastAPI application entry point.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.api import routes, websocket
from app.config import get_settings
from app.database import init_db

settings = get_settings()
templates = Jinja2Templates(directory="app/templates")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan events.

    Handles startup and shutdown tasks.
    """
    # Startup
    init_db()
    yield
    # Shutdown (cleanup if needed)


# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    description="Self-hosted audio translation service using Whisper, Claude, and Piper TTS",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS Configuration
# Allow all origins for development, restrict in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files (for voice samples, frontend assets, etc.)
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Include API routes
app.include_router(routes.router, prefix="/api")
app.include_router(websocket.router)


@app.get("/health")
async def health_check() -> dict[str, str]:
    """
    Health check endpoint.

    Returns:
        Status message
    """
    return {"status": "healthy", "version": "0.1.0"}


@app.get("/", response_class=HTMLResponse)
async def root(request: Request) -> HTMLResponse:
    """
    Root endpoint - serves the main UI.

    Args:
        request: FastAPI request object

    Returns:
        HTML page with the UI
    """
    return templates.TemplateResponse("index.html", {"request": request})
