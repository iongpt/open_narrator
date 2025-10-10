"""
OpenNarrator - Self-hosted audio translation service.

Main FastAPI application entry point.
"""

import logging
import sys
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.api import routes, websocket
from app.config import VERSION, get_settings
from app.database import init_db

templates = Jinja2Templates(directory="app/templates")

settings = get_settings()


def configure_logging() -> None:
    """
    Configure application logging based on settings.

    Sets up:
    - Log level from environment variable (DEBUG/INFO/WARNING/ERROR)
    - Detailed format with timestamp, logger name, level, and message
    - Colored output for better readability
    - Debug mode includes full request/response logging
    """
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    # Define log format based on debug mode
    if settings.debug:
        # More verbose format for debugging
        log_format = (
            "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s"
        )
    else:
        # Simpler format for production
        log_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format=log_format,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Set specific log levels for noisy third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)

    # Optionally silence SQLAlchemy (except errors)
    if settings.silence_sqlalchemy:
        logging.getLogger("sqlalchemy.engine").setLevel(logging.ERROR)
        logging.getLogger("sqlalchemy.pool").setLevel(logging.ERROR)
        logging.getLogger("sqlalchemy.dialects").setLevel(logging.ERROR)
        logging.getLogger("sqlalchemy.orm").setLevel(logging.ERROR)

    # Log startup message
    logger = logging.getLogger(__name__)
    logger.info(
        f"Logging configured: level={settings.log_level.upper()}, debug={settings.debug}, sqlalchemy_silenced={settings.silence_sqlalchemy}"
    )


configure_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan events.

    Handles startup and shutdown tasks.
    """
    # Startup
    logger.info("Starting OpenNarrator application...")
    logger.info(f"Device: {settings.device}, Compute Type: {settings.compute_type}")
    logger.info(
        f"Whisper Model: {settings.whisper_model}, Translation Model: {settings.translation_model}"
    )
    init_db()
    logger.info("Database initialized successfully")
    yield
    # Shutdown (cleanup if needed)
    logger.info("Shutting down OpenNarrator application...")


# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    description="Self-hosted audio translation service using Whisper, Claude, and Piper TTS",
    version=VERSION,
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
    return {"status": "healthy", "version": VERSION}


@app.get("/", response_class=HTMLResponse)
async def root(request: Request) -> HTMLResponse:
    """
    Root endpoint - serves the main UI.

    Args:
        request: FastAPI request object

    Returns:
        HTML page with the UI
    """
    return templates.TemplateResponse("index.html", {"request": request, "version": VERSION})
