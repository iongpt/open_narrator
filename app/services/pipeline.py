"""End-to-end audio translation pipeline orchestration.

This module orchestrates the complete STT → Translation → TTS workflow,
with progress tracking, error handling, and database updates at each stage.
"""

import asyncio
import logging
from collections.abc import Coroutine
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.api.websocket import send_progress_update
from app.config import get_settings
from app.database import SessionLocal
from app.models import Job, JobStatus
from app.schemas import ProgressUpdate
from app.services.stt_service import get_stt_service
from app.services.text_preprocessor import TextPreprocessor
from app.services.translation_service import TranslationService
from app.services.tts_service import TTSService

logger = logging.getLogger(__name__)
settings = get_settings()


def _schedule_progress_task(coro: Coroutine[Any, Any, None], job_id: int, stage: str) -> None:
    """Run an SSE update without hiding exceptions from the scheduler."""

    try:
        task: asyncio.Task[None] = asyncio.create_task(coro)
    except RuntimeError:
        logger.warning(
            "Cannot send progress update: no running event loop (job %s, stage %s)",
            job_id,
            stage,
        )
        return

    def _log_result(completed: asyncio.Task[None]) -> None:
        if completed.cancelled():
            logger.warning("Progress update task cancelled (job %s, stage %s)", job_id, stage)
            return
        exc = completed.exception()
        if exc is not None:
            logger.error(
                "Progress update task failed (job %s, stage %s): %s",
                job_id,
                stage,
                exc,
            )

    task.add_done_callback(_log_result)


async def process_audio(
    job_id: int,
    file_path: str,
    source_lang: str,
    target_lang: str,
    voice_id: str,
    context: str = "",
    file_type: str = "audio",
    skip_translation: bool = False,
    length_scale: float | None = None,
    noise_scale: float | None = None,
    noise_w_scale: float | None = None,
) -> None:
    """
    Process an audio or text file through the complete translation pipeline.

    This function orchestrates the entire workflow:
    - For audio files:
        1. Stage 1: Transcription (0-30% progress)
        2. Stage 2: Translation (30-70% progress)
        3. Stage 3: Audio Generation (70-95% progress)
        4. Stage 4: Finalization (95-100% progress)
    - For text files:
        1. Stage 1: Text Extraction (0-30% progress)
        2. Stage 2: Translation (30-70% progress)
        3. Stage 3: Audio Generation (70-95% progress)
        4. Stage 4: Finalization (95-100% progress)
    - For text files with skip_translation:
        1. Stage 1: Text Extraction (0-30% progress)
        2. Stage 2: Audio Generation (30-95% progress)
        3. Stage 3: Finalization (95-100% progress)

    Progress updates are sent via SSE after each stage, and the job is
    updated in the database with partial results (transcript, translation).

    Args:
        job_id: Database ID of the job to process
        file_path: Path to the uploaded audio or text file
        source_lang: Source language code (e.g., "en")
        target_lang: Target language code (e.g., "ro")
        voice_id: Voice ID for TTS generation
        context: Optional context for translation (e.g., "mystery novel")
        file_type: Type of file - "audio" or "text" (default: "audio")
        skip_translation: Skip translation stage (content already in target language)
        length_scale: Optional tempo override for TTS
        noise_scale: Optional expressiveness override for TTS
        noise_w_scale: Optional phoneme-width variation override for TTS

    Raises:
        None: All exceptions are caught and saved to job.error_message
    """
    # Create a new database session for this background task
    db = SessionLocal()
    preprocessor = TextPreprocessor()

    try:
        logger.info(f"Starting pipeline processing for job {job_id} (type: {file_type})")

        # Fetch job from database
        job = db.query(Job).filter(Job.id == job_id).first()
        if not job:
            logger.error(f"Job {job_id} not found in database")
            return

        # ============================================================
        # STAGE 1: TRANSCRIPTION / TEXT EXTRACTION (0-30% progress)
        # ============================================================
        try:
            if file_type == "text":
                logger.info(f"[Job {job_id}] Stage 1: Extracting text from file")

                # Update job status
                job.status = (
                    JobStatus.TRANSCRIBING
                )  # Reusing transcribing status for text extraction
                job.progress = 0.0
                db.commit()
                db.refresh(job)

                # Send progress update via SSE
                await send_progress_update(
                    ProgressUpdate(
                        job_id=job_id,
                        status=JobStatus.TRANSCRIBING,
                        progress=0.0,
                        message="Extracting text from file...",
                    )
                )

                # Initialize text extraction service and extract text
                from app.services.text_extraction_service import get_text_extraction_service

                text_service = get_text_extraction_service()
                transcript = await asyncio.to_thread(
                    text_service.extract_text,
                    file_path=file_path,
                )

                logger.info(
                    f"[Job {job_id}] Text extraction complete: {len(transcript)} characters"
                )

            else:  # audio file
                logger.info(f"[Job {job_id}] Stage 1: Starting transcription")

                # Update job status
                job.status = JobStatus.TRANSCRIBING
                job.progress = 0.0
                db.commit()
                db.refresh(job)

                # Send progress update via SSE
                await send_progress_update(
                    ProgressUpdate(
                        job_id=job_id,
                        status=JobStatus.TRANSCRIBING,
                        progress=0.0,
                        message="Starting transcription...",
                    )
                )

                # Capture event loop before entering worker thread
                loop = asyncio.get_running_loop()

                # Create progress callback for STT
                def stt_progress_callback(segments_processed: int) -> None:
                    """Update progress during transcription (called from worker thread)."""
                    # Map segment progress to 0-30% range
                    # We don't know total segments upfront, so show incremental progress
                    # Cap at 28% to leave room for finalization
                    progress = min(28.0, segments_processed * 0.5)  # ~0.5% per segment

                    # Schedule DB update on main thread via call_soon_threadsafe
                    def update_db_and_send_sse() -> None:
                        """Runs on main event loop thread."""
                        try:
                            # job is guaranteed to be non-None at this point
                            assert job is not None, "Job should not be None in callback"
                            # Refresh job to avoid stale data
                            db.refresh(job)
                            job.progress = progress
                            db.commit()

                            # Schedule SSE update as async task (now safe - we're on main loop)
                            asyncio.create_task(
                                send_progress_update(
                                    ProgressUpdate(
                                        job_id=job_id,
                                        status=JobStatus.TRANSCRIBING,
                                        progress=progress,
                                        message=f"Transcribing audio... ({segments_processed} segments processed)",
                                    )
                                )
                            )
                        except Exception as e:
                            logger.error(f"Failed to update progress during transcription: {e}")

                    # Schedule on main thread to avoid cross-thread DB access
                    loop.call_soon_threadsafe(update_db_and_send_sse)

                # Initialize STT service and transcribe
                stt_service = get_stt_service()
                transcript = await stt_service.transcribe(
                    file_path=file_path,
                    language=source_lang,
                    progress_callback=stt_progress_callback,
                )

                logger.info(f"[Job {job_id}] Transcription complete: {len(transcript)} characters")

            # Save transcript to database
            job.transcript = transcript
            job.status = JobStatus.TRANSCRIBED
            job.progress = 30.0
            db.commit()
            db.refresh(job)

            # Save transcript to debug file if debug mode is enabled
            if settings.debug:
                debug_transcript_path = settings.debug_dir / f"job{job_id}_transcript.txt"
                with open(debug_transcript_path, "w", encoding="utf-8") as f:
                    f.write(transcript)
                logger.info(f"Saved transcript to debug file: {debug_transcript_path}")

            # Send progress update
            await send_progress_update(
                ProgressUpdate(
                    job_id=job_id,
                    status=JobStatus.TRANSCRIBED,
                    progress=30.0,
                    message="Transcription complete",
                )
            )

        except Exception as e:
            logger.error(f"[Job {job_id}] Transcription failed: {str(e)}")
            await _handle_job_failure(
                db,
                job,
                f"Transcription failed: {str(e)}",
                JobStatus.TRANSCRIBING,
            )
            return

        # ============================================================
        # STAGE 2: TRANSLATION (30-70% progress)
        # ============================================================
        # Skip translation if content is already in target language
        if skip_translation:
            logger.info(
                f"[Job {job_id}] Stage 2: Skipping translation (content already in target language)"
            )
            # Use transcript as the final text (no translation needed)
            translation = transcript

            # Update job with the "translation" (which is just the original text)
            job.translation = translation
            job.status = JobStatus.TRANSLATED
            job.progress = 70.0
            db.commit()
            db.refresh(job)

            # Send progress update
            await send_progress_update(
                ProgressUpdate(
                    job_id=job_id,
                    status=JobStatus.TRANSLATED,
                    progress=70.0,
                    message="Translation skipped (content already in target language)",
                )
            )
        else:
            # Normal translation flow
            try:
                logger.info(f"[Job {job_id}] Stage 2: Starting translation")

                # Update job status
                job.status = JobStatus.TRANSLATING
                job.progress = 30.0
                db.commit()
                db.refresh(job)

                # Send progress update
                await send_progress_update(
                    ProgressUpdate(
                        job_id=job_id,
                        status=JobStatus.TRANSLATING,
                        progress=30.0,
                        message="Starting translation...",
                    )
                )

                # Initialize translation service
                translation_service = TranslationService()

                # Capture event loop for translation callbacks
                loop = asyncio.get_running_loop()

                # Create progress callback for translation chunks
                def translation_progress_callback(
                    current_chunk: int, total_chunks: int, message: str
                ) -> None:
                    """Update progress during translation."""
                    # Map chunk progress to 30-70% range
                    chunk_progress = (current_chunk / total_chunks) if total_chunks > 0 else 0
                    progress = 30.0 + (chunk_progress * 40.0)

                    # Schedule DB update on main thread via call_soon_threadsafe
                    def update_db_and_send_sse() -> None:
                        """Runs on main event loop thread."""
                        try:
                            # job is guaranteed to be non-None at this point
                            assert job is not None, "Job should not be None in callback"
                            # Refresh job to avoid stale data
                            db.refresh(job)
                            job.progress = progress
                            db.commit()

                            # Schedule SSE update as async task
                            asyncio.create_task(
                                send_progress_update(
                                    ProgressUpdate(
                                        job_id=job_id,
                                        status=JobStatus.TRANSLATING,
                                        progress=progress,
                                        message=message,
                                    )
                                )
                            )
                        except Exception as e:
                            logger.error(f"Failed to update progress during translation: {e}")

                    # Schedule on main thread
                    loop.call_soon_threadsafe(update_db_and_send_sse)

                # Translate text
                translation = await translation_service.translate(
                    text=transcript,
                    source_lang=source_lang,
                    target_lang=target_lang,
                    context=context,
                    progress_callback=translation_progress_callback,
                )

                logger.info(f"[Job {job_id}] Translation complete: {len(translation)} characters")

                # Save translation to database
                job.translation = translation
                job.status = JobStatus.TRANSLATED
                job.progress = 70.0
                db.commit()
                db.refresh(job)

                # Save translation to debug file if debug mode is enabled
                if settings.debug:
                    debug_translation_path = settings.debug_dir / f"job{job_id}_translation.txt"
                    with open(debug_translation_path, "w", encoding="utf-8") as f:
                        f.write(translation)
                    logger.info(f"Saved translation to debug file: {debug_translation_path}")

                # Send progress update
                await send_progress_update(
                    ProgressUpdate(
                        job_id=job_id,
                        status=JobStatus.TRANSLATED,
                        progress=70.0,
                        message="Translation complete",
                    )
                )

            except Exception as e:
                logger.error(f"[Job {job_id}] Translation failed: {str(e)}")
                await _handle_job_failure(
                    db,
                    job,
                    f"Translation failed: {str(e)}",
                    JobStatus.TRANSLATING,
                )
                return

        # ============================================================
        # STAGE 3: AUDIO GENERATION (70-95% progress)
        # ============================================================
        try:
            logger.info(f"[Job {job_id}] Stage 3: Starting audio generation")

            # Update job status
            job.status = JobStatus.GENERATING_AUDIO
            job.progress = 70.0
            db.commit()
            db.refresh(job)

            # Send progress update
            await send_progress_update(
                ProgressUpdate(
                    job_id=job_id,
                    status=JobStatus.GENERATING_AUDIO,
                    progress=70.0,
                    message="Generating audio...",
                )
            )

            # Initialize TTS service
            tts_service = TTSService()

            # Capture event loop for TTS callbacks
            loop = asyncio.get_running_loop()

            # Create progress callback for TTS
            def tts_progress_callback(tts_progress: float) -> None:
                """Update progress during TTS generation."""
                # Map TTS progress (0.0-1.0) to 70-95% range
                progress = 70.0 + (tts_progress * 25.0)

                # Display percentage with a single decimal so long jobs surface progress
                message = f"Generating audio... {tts_progress * 100:.1f}%"
                if tts_progress >= 1.0:
                    message = "Generating audio... finalizing"

                # Schedule DB update on main thread via call_soon_threadsafe
                def update_db_and_send_sse() -> None:
                    """Runs on main event loop thread."""
                    try:
                        # job is guaranteed to be non-None at this point
                        assert job is not None, "Job should not be None in callback"
                        # Refresh job to avoid stale data
                        db.refresh(job)
                        job.progress = progress
                        db.commit()

                        # Schedule SSE update as async task
                        asyncio.create_task(
                            send_progress_update(
                                ProgressUpdate(
                                    job_id=job_id,
                                    status=JobStatus.GENERATING_AUDIO,
                                    progress=progress,
                                    message=message,
                                )
                            )
                        )
                    except Exception as e:
                        logger.error(f"Failed to update progress during TTS: {e}")

                # Schedule on main thread
                loop.call_soon_threadsafe(update_db_and_send_sse)

            # Generate audio
            prepared_translation = preprocessor.prepare_for_tts(translation)
            output_path = await tts_service.generate_audio(
                text=prepared_translation,
                voice_id=voice_id,
                language=target_lang,
                progress_callback=tts_progress_callback,
                job_id=job.id,
                length_scale=job.length_scale,
                noise_scale=job.noise_scale,
                noise_w_scale=job.noise_w_scale,
            )

            logger.info(f"[Job {job_id}] Audio generation complete: {output_path}")

            # Verify output file exists
            if not Path(output_path).exists():
                raise RuntimeError(f"Output file not found: {output_path}")

            # Save output path to database
            job.output_path = output_path
            job.progress = 95.0
            db.commit()
            db.refresh(job)

            # Send progress update
            await send_progress_update(
                ProgressUpdate(
                    job_id=job_id,
                    status=JobStatus.GENERATING_AUDIO,
                    progress=95.0,
                    message="Audio generation complete",
                )
            )

        except Exception as e:
            logger.error(f"[Job {job_id}] Audio generation failed: {str(e)}")
            await _handle_job_failure(
                db,
                job,
                f"Audio generation failed: {str(e)}",
                JobStatus.GENERATING_AUDIO,
            )
            return

        # ============================================================
        # STAGE 4: FINALIZATION (95-100% progress)
        # ============================================================
        try:
            logger.info(f"[Job {job_id}] Stage 4: Finalizing job")

            # Mark job as completed
            job.status = JobStatus.COMPLETED
            job.progress = 100.0
            job.completed_at = datetime.utcnow()
            db.commit()
            db.refresh(job)

            # Send final progress update
            await send_progress_update(
                ProgressUpdate(
                    job_id=job_id,
                    status=JobStatus.COMPLETED,
                    progress=100.0,
                    message="Processing complete! Ready for download.",
                )
            )

            logger.info(f"[Job {job_id}] Pipeline processing completed successfully")

            # Optional: Clean up temporary files
            try:
                original_file = Path(file_path)
                if original_file.exists():
                    if original_file.suffix == ".wav" and original_file.name.startswith("tmp"):
                        logger.info(f"Cleaning up temporary file: {original_file}")
                        original_file.unlink()
                    else:
                        resolved_upload_dir = settings.upload_dir.resolve()
                        if resolved_upload_dir in original_file.resolve().parents:
                            logger.info(f"Removing original upload file: {original_file}")
                            original_file.unlink()
            except Exception as e:
                logger.warning(f"Failed to clean up uploaded file: {str(e)}")

        except Exception as e:
            logger.error(f"[Job {job_id}] Finalization failed: {str(e)}")
            # Even if finalization fails, the job is mostly complete
            # Just log the error and update the status
            job.error_message = f"Finalization warning: {str(e)}"
            db.commit()

    except Exception as e:
        # Catch-all for any unexpected errors
        logger.error(f"[Job {job_id}] Unexpected error in pipeline: {str(e)}")
        try:
            job = db.query(Job).filter(Job.id == job_id).first()
            if job:
                await _handle_job_failure(
                    db,
                    job,
                    f"Unexpected error: {str(e)}",
                    job.status,
                )
        except Exception as e2:
            logger.error(f"Failed to update job status after error: {str(e2)}")

    finally:
        # Always close the database session
        db.close()


async def _handle_job_failure(
    db: Session,
    job: Job,
    error_message: str,
    failed_stage: JobStatus,
) -> None:
    """
    Handle job failure by updating status and sending notification.

    Args:
        db: Database session
        job: The job that failed
        error_message: Error message to save
        failed_stage: The stage where the job failed
    """
    try:
        # Update job with error
        job.status = JobStatus.FAILED
        job.error_message = error_message
        db.commit()
        db.refresh(job)

        # Send failure notification via SSE
        await send_progress_update(
            ProgressUpdate(
                job_id=job.id,
                status=JobStatus.FAILED,
                progress=job.progress,
                message=f"Failed at {failed_stage.value} stage: {error_message}",
            )
        )

        logger.error(f"[Job {job.id}] Job failed: {error_message}")

    except Exception as e:
        logger.error(f"Failed to handle job failure: {str(e)}")


def cleanup_failed_job_files(job: Job) -> None:
    """
    Clean up files associated with a failed job.

    Args:
        job: The failed job to clean up
    """
    try:
        # Delete original file
        if job.original_path:
            original_path = Path(job.original_path)
            if original_path.exists():
                original_path.unlink()
                logger.info(f"Deleted original file: {original_path}")

        # Delete partial output file (if exists)
        if job.output_path:
            output_path = Path(job.output_path)
            if output_path.exists():
                output_path.unlink()
                logger.info(f"Deleted partial output file: {output_path}")

    except Exception as e:
        logger.warning(f"Failed to clean up files for job {job.id}: {str(e)}")
