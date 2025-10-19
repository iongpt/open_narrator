"""Background worker that watches a folder for new bulk jobs."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.websocket import send_progress_update
from app.config import get_settings
from app.constants import ALLOWED_AUDIO_EXTENSIONS, ALLOWED_TEXT_EXTENSIONS
from app.database import SessionLocal
from app.models import Job, JobStatus
from app.schemas import ProgressUpdate
from app.services.bulk_preset import ensure_output_directory, load_bulk_preset

logger = logging.getLogger(__name__)

_ALLOWED_EXTENSIONS = ALLOWED_TEXT_EXTENSIONS | ALLOWED_AUDIO_EXTENSIONS

_ACTIVE_STATUSES = {
    JobStatus.PENDING,
    JobStatus.DISPATCHING,
    JobStatus.TRANSCRIBING,
    JobStatus.TRANSCRIBED,
    JobStatus.TRANSLATING,
    JobStatus.TRANSLATED,
    JobStatus.GENERATING_AUDIO,
}


class BulkIngestWorker:
    """Simple polling worker that creates jobs from a monitored folder."""

    def __init__(
        self,
        *,
        session_factory: Callable[[], Session] | None = None,
        poll_interval: float | None = None,
    ) -> None:
        settings = get_settings()
        self._session_factory = session_factory or SessionLocal
        self._poll_interval = (
            poll_interval if poll_interval is not None else settings.bulk_scan_interval_seconds
        )

        self._shutdown_event = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """Launch the polling loop."""

        if self._task is not None:
            logger.debug("BulkIngestWorker already running")
            return

        self._shutdown_event.clear()
        self._task = asyncio.create_task(self._run_loop(), name="bulk-ingest-worker")
        logger.info("Started bulk ingest worker")

    async def stop(self) -> None:
        """Stop the polling loop."""

        if self._task is None:
            return

        self._shutdown_event.set()
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        finally:
            self._task = None
            logger.info("Stopped bulk ingest worker")

    async def _run_loop(self) -> None:
        try:
            while not self._shutdown_event.is_set():
                try:
                    await self._poll_once()
                except Exception:  # pragma: no cover - defensive log
                    logger.exception("Bulk ingest poll failed")

                try:
                    await asyncio.wait_for(self._shutdown_event.wait(), timeout=self._poll_interval)
                except TimeoutError:
                    continue
        except asyncio.CancelledError:  # pragma: no cover
            raise

    async def _poll_once(self) -> None:
        preset = load_bulk_preset()
        if not preset:
            logger.debug("Bulk ingest preset not configured yet")
            return

        if not preset.voice_id or not preset.target_language:
            logger.debug("Bulk preset missing voice_id or target_language; skipping poll")
            return

        input_root = Path(preset.input_dir).expanduser().resolve()
        if not input_root.exists():
            logger.debug("Bulk input dir does not exist: %s", input_root)
            return

        ensure_output_directory(preset.output_dir)
        output_root = Path(preset.output_dir).expanduser().resolve()

        candidate_files = self._discover_files(input_root)
        if not candidate_files:
            logger.debug("No new files discovered in bulk input directory")
            return

        session = self._session_factory()
        try:
            for file_path in candidate_files:
                try:
                    if self._has_active_job(session, file_path):
                        continue

                    file_type = self._infer_file_type(file_path)
                    skip_translation = preset.skip_translation and file_type == "text"
                    if file_type == "audio" and preset.skip_translation:
                        logger.warning(
                            "Skipping %s because skip_translation preset is incompatible with audio files",
                            file_path,
                        )
                        continue

                    relative_path = file_path.relative_to(input_root)
                    target_output = (output_root / relative_path).with_suffix(".mp3")
                    target_output.parent.mkdir(parents=True, exist_ok=True)

                    job = Job(
                        filename=str(relative_path),
                        original_path=str(file_path),
                        source_language=preset.source_language,
                        target_language=preset.target_language,
                        voice_id=preset.voice_id,
                        context=preset.context,
                        skip_translation=skip_translation,
                        length_scale=preset.length_scale,
                        noise_scale=preset.noise_scale,
                        noise_w_scale=preset.noise_w_scale,
                        status=JobStatus.PENDING,
                        progress=0.0,
                        cleanup_original=True,
                        target_output_path=str(target_output),
                    )

                    session.add(job)
                    session.commit()
                    session.refresh(job)

                    logger.info(
                        "Queued bulk job %s for %s → %s",
                        job.id,
                        job.source_language,
                        job.target_language,
                    )

                    await send_progress_update(
                        ProgressUpdate(
                            job_id=job.id,
                            status=job.status,
                            progress=job.progress,
                            message=f"Queued from bulk ingest: {relative_path}",
                        )
                    )
                except Exception:
                    logger.exception("Failed to queue bulk job for file %s", file_path)
                    session.rollback()
        finally:
            session.close()

    @staticmethod
    def _infer_file_type(path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix in ALLOWED_TEXT_EXTENSIONS:
            return "text"
        if suffix in ALLOWED_AUDIO_EXTENSIONS:
            return "audio"
        return "audio"  # Default fallback

    def _discover_files(self, input_root: Path) -> list[Path]:
        files: list[Path] = []
        for path in input_root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in _ALLOWED_EXTENSIONS:
                continue
            files.append(path)
        return sorted(files)

    def _has_active_job(self, session: Session, file_path: Path) -> bool:
        stmt = (
            select(Job.id)
            .where(Job.original_path == str(file_path))
            .where(Job.status.in_(_ACTIVE_STATUSES))
        )
        return session.execute(stmt).scalar_one_or_none() is not None
