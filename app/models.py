"""Database models for OpenNarrator."""

from datetime import datetime
from enum import Enum as PyEnum

from sqlalchemy import Enum, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class JobStatus(str, PyEnum):
    """Job processing status."""

    PENDING = "pending"
    TRANSCRIBING = "transcribing"
    TRANSCRIBED = "transcribed"
    TRANSLATING = "translating"
    TRANSLATED = "translated"
    GENERATING_AUDIO = "generating_audio"
    COMPLETED = "completed"
    FAILED = "failed"


class Job(Base):
    """
    Audio translation job.

    Tracks the entire pipeline: upload → transcription → translation → TTS → download
    """

    __tablename__ = "jobs"

    # Primary key
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)

    # File information
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    original_path: Mapped[str] = mapped_column(String(500), nullable=False)
    output_path: Mapped[str | None] = mapped_column(String(500), nullable=True)

    # Language settings
    source_language: Mapped[str] = mapped_column(String(10), default="en")
    target_language: Mapped[str] = mapped_column(String(10), nullable=False)

    # Voice settings
    voice_id: Mapped[str] = mapped_column(String(100), nullable=False)

    # Context for translation
    context: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Processing status
    status: Mapped[JobStatus] = mapped_column(
        Enum(JobStatus), default=JobStatus.PENDING, nullable=False, index=True
    )
    progress: Mapped[float] = mapped_column(Float, default=0.0)  # 0.0 to 100.0

    # Intermediate results (stored for debugging/resume)
    transcript: Mapped[str | None] = mapped_column(Text, nullable=True)
    translation: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Error handling
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Metadata
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )
    completed_at: Mapped[datetime | None] = mapped_column(nullable=True)

    def __repr__(self) -> str:
        """String representation of Job."""
        return f"<Job(id={self.id}, filename={self.filename}, status={self.status})>"
