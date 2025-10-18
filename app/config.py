"""Application configuration and settings."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

import torch
from pydantic_settings import BaseSettings, SettingsConfigDict

# Application version - single source of truth
VERSION = "0.3.1"


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        protected_namespaces=(
            "settings_",
        ),  # Change from default 'model_' to avoid conflict with model_dir
    )

    # Application
    app_name: str = "OpenNarrator"
    debug: bool = False
    log_level: str = "INFO"
    silence_sqlalchemy: bool = True  # Silence SQLAlchemy logs except errors

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1

    # Database
    database_url: str = "sqlite:///./data/app.db"

    # File Storage
    upload_dir: Path = Path("./data/uploads")
    output_dir: Path = Path("./data/outputs")
    model_dir: Path = Path("./data/models")
    debug_dir: Path = Path("./data/debug")  # Debug files (transcripts, translations)
    static_dir: Path = Path("./app/static")  # Static files (voice samples, etc.)
    max_upload_size_mb: int = 50

    # AI Services
    anthropic_api_key: str = ""

    # Whisper Settings
    whisper_model: Literal["tiny", "base", "small", "medium", "large-v2", "large-v3"] = "large-v3"
    whisper_compute_type: Literal["auto", "int8", "int8_float16", "int8_float32", "float16"] = (
        "auto"
    )
    whisper_vad_filter: bool = False  # Voice Activity Detection - can cause truncation issues

    # Translation Settings
    translation_model: str = "claude-sonnet-4.5-20250514"
    translation_max_tokens: int = 20000  # Max tokens per chunk (input)
    translation_max_output_tokens: int = 64000  # Max output tokens from LLM

    # TTS Settings
    tts_engine: Literal["piper"] = "piper"

    @property
    def device(self) -> str:
        """Auto-detect and return compute device (cuda or cpu)."""
        return "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def compute_type(self) -> str:
        """
        Determine optimal compute type based on device.

        Returns:
            - For GPU: float16 (fastest with good quality)
            - For CPU: int8 (best CPU performance)
            - If manual override: use whisper_compute_type
        """
        if self.whisper_compute_type != "auto":
            return self.whisper_compute_type

        if self.device == "cuda":
            return "float16"
        return "int8"

    def ensure_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        self.static_dir.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Returns:
        Singleton Settings instance
    """
    settings = Settings()
    settings.ensure_directories()
    return settings
