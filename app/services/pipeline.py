"""End-to-end audio translation pipeline orchestration.

This module orchestrates the complete STT → Translation → TTS workflow,
with progress tracking, error handling, and database updates at each stage.
"""

import logging
from datetime import datetime
from pathlib import Path

from sqlalchemy.orm import Session

from app.api.websocket import send_progress_update
from app.config import get_settings
from app.database import SessionLocal
from app.models import Job, JobStatus
from app.schemas import ProgressUpdate
from app.services.stt_service import STTService
from app.services.translation_service import TranslationService
from app.services.tts_service import TTSService

logger = logging.getLogger(__name__)
settings = get_settings()


async def process_audio(
    job_id: int,
    file_path: str,
    source_lang: str,
    target_lang: str,
    voice_id: str,
    context: str = "",
) -> None:
    """
    Process an audio file through the complete translation pipeline.

    This function orchestrates the entire workflow:
    1. Stage 1: Transcription (0-30% progress)
    2. Stage 2: Translation (30-70% progress)
    3. Stage 3: Audio Generation (70-95% progress)
    4. Stage 4: Finalization (95-100% progress)

    Progress updates are sent via SSE after each stage, and the job is
    updated in the database with partial results (transcript, translation).

    Args:
        job_id: Database ID of the job to process
        file_path: Path to the uploaded audio file
        source_lang: Source language code (e.g., "en")
        target_lang: Target language code (e.g., "ro")
        voice_id: Voice ID for TTS generation
        context: Optional context for translation (e.g., "mystery novel")

    Raises:
        None: All exceptions are caught and saved to job.error_message
    """
    # Create a new database session for this background task
    db = SessionLocal()

    try:
        logger.info(f"Starting pipeline processing for job {job_id}")

        # Fetch job from database
        job = db.query(Job).filter(Job.id == job_id).first()
        if not job:
            logger.error(f"Job {job_id} not found in database")
            return

        # ============================================================
        # STAGE 1: TRANSCRIPTION (0-30% progress)
        # ============================================================
        try:
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

            # Initialize STT service and transcribe
            stt_service = STTService()
            transcript = await stt_service.transcribe(
                file_path=file_path,
                language=source_lang,
            )

            logger.info(f"[Job {job_id}] Transcription complete: {len(transcript)} characters")

            # Save transcript to database
            job.transcript = transcript
            job.status = JobStatus.TRANSCRIBED
            job.progress = 30.0
            db.commit()
            db.refresh(job)

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

            # Create progress callback for translation chunks
            def translation_progress_callback(
                current_chunk: int, total_chunks: int, message: str
            ) -> None:
                """Update progress during translation."""
                # Map chunk progress to 30-70% range
                chunk_progress = (current_chunk / total_chunks) if total_chunks > 0 else 0
                progress = 30.0 + (chunk_progress * 40.0)

                # Update database (job is guaranteed to exist at this point)
                assert job is not None, "Job should not be None in callback"
                job.progress = progress
                db.commit()

                # Send SSE update (non-blocking)
                import asyncio

                try:
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
                except RuntimeError:
                    # If no event loop is running, log warning
                    logger.warning("Cannot send progress update: no event loop")

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

            # Create progress callback for TTS
            def tts_progress_callback(tts_progress: float) -> None:
                """Update progress during TTS generation."""
                # Map TTS progress (0.0-1.0) to 70-95% range
                progress = 70.0 + (tts_progress * 25.0)

                # Update database (job is guaranteed to exist at this point)
                assert job is not None, "Job should not be None in callback"
                job.progress = progress
                db.commit()

                # Send SSE update (non-blocking)
                import asyncio

                try:
                    asyncio.create_task(
                        send_progress_update(
                            ProgressUpdate(
                                job_id=job_id,
                                status=JobStatus.GENERATING_AUDIO,
                                progress=progress,
                                message=f"Generating audio... {int(tts_progress * 100)}%",
                            )
                        )
                    )
                except RuntimeError:
                    logger.warning("Cannot send progress update: no event loop")

            # Generate audio
            output_path = await tts_service.generate_audio(
                text=translation,
                voice_id=voice_id,
                language=target_lang,
                progress_callback=tts_progress_callback,
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
                if original_file.exists() and original_file.suffix == ".wav":
                    # Only delete temporary WAV files (converted from MP3)
                    logger.info(f"Cleaning up temporary file: {original_file}")
                    original_file.unlink()
            except Exception as e:
                logger.warning(f"Failed to clean up temporary file: {str(e)}")

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
