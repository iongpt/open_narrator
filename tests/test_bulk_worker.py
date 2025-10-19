"""Tests for the bulk folder ingestion worker."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import get_settings
from app.database import Base
from app.models import Job, JobStatus
from app.schemas import BulkPreset
from app.services.bulk_preset import save_bulk_preset
from app.services.bulk_worker import BulkIngestWorker


@pytest.fixture()
def session_factory(tmp_path: Path) -> sessionmaker[Session]:
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine, autocommit=False, autoflush=False)


@pytest.mark.asyncio
async def test_bulk_worker_creates_jobs(
    tmp_path: Path,
    session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = get_settings()

    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    preset_path = tmp_path / "preset.json"

    input_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)

    monkeypatch.setattr(settings, "bulk_input_dir", input_dir)
    monkeypatch.setattr(settings, "bulk_output_dir", output_dir)
    monkeypatch.setattr(settings, "bulk_preset_path", preset_path)

    preset = BulkPreset(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        source_language="en",
        target_language="ro",
        voice_id="piper:en_US-lessac-medium",
        context="Epic fantasy",
        skip_translation=False,
    )
    save_bulk_preset(preset)

    sample_file = input_dir / "Author" / "Saga" / "Book One.epub"
    sample_file.parent.mkdir(parents=True, exist_ok=True)
    sample_file.write_text("dummy", encoding="utf-8")

    worker = BulkIngestWorker(session_factory=session_factory)
    await worker._poll_once()

    session = session_factory()
    jobs = session.query(Job).all()
    session.close()

    assert len(jobs) == 1
    job = jobs[0]
    assert job.filename.endswith("Book One.epub")
    assert job.original_path == str(sample_file)
    assert job.target_output_path == str(output_dir / "Author" / "Saga" / "Book One.mp3")
    assert job.cleanup_original is True
    assert job.status == JobStatus.PENDING

    # Second poll should not duplicate the job while the original is pending
    await worker._poll_once()
    session = session_factory()
    assert session.query(Job).count() == 1
    session.close()
