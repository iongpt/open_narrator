"""Database-backed dispatcher that schedules translation jobs.

This module introduces a lightweight job runner that keeps the upload
queue durable inside the existing `jobs` table. The dispatcher runs as a
long-lived asyncio task, pulls pending jobs from the database, and feeds
them through the existing pipeline while respecting a configurable
concurrency limit.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.api.websocket import send_progress_update
from app.config import Settings, get_settings
from app.database import SessionLocal
from app.models import Job, JobStatus
from app.schemas import ProgressUpdate

logger = logging.getLogger(__name__)

# Text extensions accepted by the upload endpoint; anything else is treated as audio.
_TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".pdf",
    ".epub",
    ".mobi",
    ".docx",
    ".doc",
    ".rtf",
    ".odt",
    ".html",
    ".htm",
}


@dataclass(slots=True, frozen=True)
class ClaimedJob:
    """Snapshot of a job that has been claimed for processing."""

    id: int
    file_path: str
    source_lang: str
    target_lang: str
    voice_id: str
    context: str
    skip_translation: bool
    length_scale: float | None
    noise_scale: float | None
    noise_w_scale: float | None


class JobDispatcher:
    """Coordinates background processing with a durable database queue."""

    def __init__(
        self,
        *,
        session_factory: Callable[[], Session] | None = None,
        settings: Settings | None = None,
        poll_interval: float | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._session_factory = session_factory or SessionLocal
        self._poll_interval = (
            poll_interval
            if poll_interval is not None
            else max(0.1, float(self._settings.dispatcher_poll_interval_seconds))
        )
        self._max_parallel = max(1, int(self._settings.max_concurrent_jobs))

        self._shutdown_event = asyncio.Event()
        self._dispatcher_task: asyncio.Task[None] | None = None
        self._active_tasks: set[asyncio.Task[None]] = set()

    async def start(self) -> None:
        """Launch the dispatcher loop if it is not already running."""
        if self._dispatcher_task is not None:
            logger.debug("JobDispatcher already running; start() call ignored")
            return

        await asyncio.to_thread(self._reset_incomplete_jobs)
        logger.info("Job dispatcher starting with max_concurrent_jobs=%s", self._max_parallel)

        self._shutdown_event.clear()
        self._dispatcher_task = asyncio.create_task(self._run_loop(), name="job-dispatcher")

    async def stop(self) -> None:
        """Signal the dispatcher loop to exit and await in-flight work."""
        if self._dispatcher_task is None:
            return

        logger.info("Stopping job dispatcher")
        self._shutdown_event.set()

        # Wait for the loop task to exit gracefully.
        try:
            await self._dispatcher_task
        finally:
            self._dispatcher_task = None

        # Cancel any remaining pipeline tasks (they will be restarted on boot).
        for task in list(self._active_tasks):
            task.cancel()
        if self._active_tasks:
            await asyncio.gather(*self._active_tasks, return_exceptions=True)
        self._active_tasks.clear()

    async def _run_loop(self) -> None:
        """Main loop: claim jobs when capacity is available."""
        try:
            while not self._shutdown_event.is_set():
                self._prune_finished_tasks()

                if len(self._active_tasks) >= self._max_parallel:
                    await self._sleep()
                    continue

                claimed = await asyncio.to_thread(self._claim_next_job)
                if claimed is None:
                    await self._sleep()
                    continue

                await send_progress_update(
                    ProgressUpdate(
                        job_id=claimed.id,
                        status=JobStatus.DISPATCHING,
                        progress=0.0,
                        message="Assigning worker...",
                    )
                )

                task = asyncio.create_task(self._execute_job(claimed), name=f"job-{claimed.id}")
                self._active_tasks.add(task)
                task.add_done_callback(self._active_tasks.discard)
        except asyncio.CancelledError:  # pragma: no cover - defensive shutdown
            raise
        finally:
            logger.info("Job dispatcher loop exited")

    async def _execute_job(self, job: ClaimedJob) -> None:
        """Run the pipeline for a claimed job and release capacity on completion."""
        file_type = self._infer_file_type(job.file_path)
        from app.services.pipeline import (
            process_audio,
        )

        try:
            await process_audio(
                job_id=job.id,
                file_path=job.file_path,
                source_lang=job.source_lang,
                target_lang=job.target_lang,
                voice_id=job.voice_id,
                context=job.context,
                file_type=file_type,
                skip_translation=job.skip_translation,
                length_scale=job.length_scale,
                noise_scale=job.noise_scale,
                noise_w_scale=job.noise_w_scale,
            )
        except asyncio.CancelledError:  # pragma: no cover - defensive shutdown
            logger.warning("Pipeline task for job %s cancelled", job.id)
            raise
        except Exception:  # pragma: no cover - pipeline already logs failures
            logger.exception("Unexpected error while processing job %s", job.id)

    def _claim_next_job(self) -> ClaimedJob | None:
        """Atomically select and lock the next pending job."""
        session = self._session_factory()
        try:
            job_id = session.execute(
                select(Job.id)
                .where(Job.status == JobStatus.PENDING)
                .order_by(Job.created_at.asc())
                .limit(1)
            ).scalar_one_or_none()

            if job_id is None:
                return None

            update_result = session.execute(
                update(Job)
                .where(Job.id == job_id, Job.status == JobStatus.PENDING)
                .values(status=JobStatus.DISPATCHING)
            )
            session.commit()

            if update_result.rowcount == 0:
                return None

            job = session.get(Job, job_id)
            if job is None:
                return None

            return ClaimedJob(
                id=job.id,
                file_path=job.original_path,
                source_lang=job.source_language,
                target_lang=job.target_language,
                voice_id=job.voice_id,
                context=job.context or "",
                skip_translation=job.skip_translation,
                length_scale=job.length_scale,
                noise_scale=job.noise_scale,
                noise_w_scale=job.noise_w_scale,
            )
        finally:
            session.close()

    def _reset_incomplete_jobs(self) -> None:
        """Return any in-flight jobs to the pending state on startup."""
        session = self._session_factory()
        try:
            affected_states = (
                JobStatus.DISPATCHING,
                JobStatus.TRANSCRIBING,
                JobStatus.TRANSCRIBED,
                JobStatus.TRANSLATING,
                JobStatus.TRANSLATED,
                JobStatus.GENERATING_AUDIO,
            )
            update_result = session.execute(
                update(Job).where(Job.status.in_(affected_states)).values(status=JobStatus.PENDING)
            )
            if update_result.rowcount:
                logger.info("Reset %s incomplete job(s) to pending state", update_result.rowcount)
            session.commit()
        finally:
            session.close()

    async def _sleep(self) -> None:
        """Sleep for the configured poll interval unless shutdown triggers."""
        if self._poll_interval <= 0:
            await asyncio.sleep(0)
            return

        try:
            await asyncio.wait_for(self._shutdown_event.wait(), timeout=self._poll_interval)
        except TimeoutError:
            return

    def _prune_finished_tasks(self) -> None:
        """Remove completed tasks from the active set to avoid memory growth."""
        for task in list(self._active_tasks):
            if task.done():
                self._active_tasks.discard(task)

    @staticmethod
    def _infer_file_type(file_path: str) -> str:
        """Best-effort inference of file type when the dispatcher claims a job."""
        suffix = Path(file_path).suffix.lower()
        return "text" if suffix in _TEXT_EXTENSIONS else "audio"


__all__ = ["JobDispatcher"]
