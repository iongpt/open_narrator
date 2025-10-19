"""Tests for the database-backed job dispatcher."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import Settings
from app.database import Base
from app.models import Job, JobStatus
from app.services.job_dispatcher import JobDispatcher


@pytest.fixture()
def in_memory_session_factory() -> sessionmaker[Session]:
    """Create an isolated in-memory SQLite session factory for dispatcher tests."""

    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine, autocommit=False, autoflush=False)


@pytest.fixture()
def dispatcher(in_memory_session_factory: sessionmaker[Session]) -> JobDispatcher:
    settings = Settings(
        max_concurrent_jobs=2,
        dispatcher_poll_interval_seconds=0.0,
    )
    return JobDispatcher(
        session_factory=in_memory_session_factory,
        settings=settings,
        poll_interval=0.0,
    )


def _create_job(session: Session, **overrides: Any) -> Job:
    payload = {
        "filename": "sample.mp3",
        "original_path": str(Path("/tmp/sample.mp3")),
        "source_language": "en",
        "target_language": "ro",
        "voice_id": "piper:en_US",
        "context": None,
        "skip_translation": False,
        "status": JobStatus.PENDING,
        "progress": 0.0,
    }
    payload.update(overrides)

    job = Job(**payload)
    session.add(job)
    session.commit()
    session.refresh(job)
    return job


def test_claim_next_job_marks_dispatching(
    dispatcher: JobDispatcher, in_memory_session_factory: sessionmaker[Session]
) -> None:
    session = in_memory_session_factory()
    job = _create_job(session)

    claimed = dispatcher._claim_next_job()

    assert claimed is not None
    assert claimed.id == job.id
    assert claimed.context == ""

    session.refresh(job)
    assert job.status == JobStatus.DISPATCHING

    session.close()


def test_reset_incomplete_jobs(
    dispatcher: JobDispatcher, in_memory_session_factory: sessionmaker[Session]
) -> None:
    session = in_memory_session_factory()
    stuck = _create_job(session, status=JobStatus.TRANSCRIBING)
    failed = _create_job(session, status=JobStatus.FAILED)

    dispatcher._reset_incomplete_jobs()

    session.refresh(stuck)
    session.refresh(failed)

    assert stuck.status == JobStatus.PENDING
    assert failed.status == JobStatus.FAILED

    session.close()


def test_infer_file_type_text() -> None:
    assert JobDispatcher._infer_file_type("/tmp/book.epub") == "text"
    assert JobDispatcher._infer_file_type("/tmp/audio.mp3") == "audio"
