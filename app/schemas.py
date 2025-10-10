"""Pydantic schemas for request/response validation."""

from datetime import datetime

from pydantic import BaseModel, Field

from app.models import JobStatus


class JobCreate(BaseModel):
    """Schema for creating a new job."""

    filename: str = Field(..., min_length=1, max_length=255)
    source_language: str = Field(default="en", min_length=2, max_length=10)
    target_language: str = Field(..., min_length=2, max_length=10)
    voice_id: str = Field(..., min_length=1, max_length=100)
    context: str | None = Field(default=None, max_length=5000)


class JobUpdate(BaseModel):
    """Schema for updating a job."""

    status: JobStatus | None = None
    progress: float | None = Field(None, ge=0.0, le=100.0)
    transcript: str | None = None
    translation: str | None = None
    output_path: str | None = None
    error_message: str | None = None


class JobResponse(BaseModel):
    """Schema for job response."""

    id: int
    filename: str
    source_language: str
    target_language: str
    voice_id: str
    context: str | None
    status: JobStatus
    progress: float
    error_message: str | None
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None
    output_path: str | None

    class Config:
        """Pydantic configuration."""

        from_attributes = True


class VoiceInfo(BaseModel):
    """Schema for voice information."""

    id: str
    name: str
    language: str
    gender: str | None = None
    quality: str | None = None
    sample_url: str | None = None


class ProgressUpdate(BaseModel):
    """Schema for SSE progress updates."""

    job_id: int
    status: JobStatus
    progress: float
    message: str | None = None


class SettingsUpdate(BaseModel):
    """Schema for updating application settings."""

    anthropic_api_key: str | None = None
    whisper_model: str | None = None
    tts_engine: str | None = None
