"""API routes for OpenNarrator."""

import os
from datetime import datetime
from pathlib import Path

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database import get_db
from app.models import Job, JobStatus
from app.schemas import JobResponse, SettingsUpdate, VoiceInfo
from app.services.pipeline import process_audio

router = APIRouter()
settings = get_settings()
templates = Jinja2Templates(directory="app/templates")


# File upload validation
ALLOWED_AUDIO_TYPES = {
    "audio/mpeg",
    "audio/mp3",
    "audio/x-mp3",
    "audio/mpeg3",
    "audio/x-mpeg-3",
}
MAX_FILE_SIZE = settings.max_upload_size_mb * 1024 * 1024  # Convert to bytes


def validate_audio_file(file: UploadFile) -> None:
    """
    Validate uploaded audio file.

    Args:
        file: The uploaded file to validate

    Raises:
        HTTPException: If file is invalid
    """
    # Check file type
    if file.content_type not in ALLOWED_AUDIO_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type. Allowed types: {', '.join(ALLOWED_AUDIO_TYPES)}",
        )

    # Check filename
    if not file.filename or not file.filename.lower().endswith(".mp3"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must have .mp3 extension",
        )


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to prevent path traversal attacks.

    Args:
        filename: Original filename

    Returns:
        Sanitized filename safe for storage
    """
    # Remove path components, keep only basename
    safe_name = os.path.basename(filename)

    # Remove any potentially dangerous characters
    safe_name = "".join(c for c in safe_name if c.isalnum() or c in "._- ")

    # Ensure non-empty
    if not safe_name:
        safe_name = "upload.mp3"

    return safe_name


@router.post("/upload", response_model=JobResponse, status_code=status.HTTP_201_CREATED)
async def upload_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    source_language: str = Form("en"),
    target_language: str = Form(...),
    voice_id: str = Form(...),
    context: str | None = Form(None),
    db: Session = Depends(get_db),
) -> Job:
    """
    Upload an MP3 file for translation.

    Args:
        background_tasks: FastAPI background tasks manager
        file: The audio file to upload
        source_language: Source language code (default: en)
        target_language: Target language code
        voice_id: Voice ID for TTS
        context: Optional context for translation
        db: Database session

    Returns:
        Created job details

    Raises:
        HTTPException: If file validation fails or upload error occurs
    """
    # Validate file
    validate_audio_file(file)

    # Sanitize filename
    safe_filename = sanitize_filename(file.filename or "upload.mp3")

    # Create unique filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_filename = f"{timestamp}_{safe_filename}"

    # Save file
    file_path = settings.upload_dir / unique_filename

    try:
        # Read and write file
        contents = await file.read()

        # Check file size
        if len(contents) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File too large. Maximum size: {settings.max_upload_size_mb}MB",
            )

        with open(file_path, "wb") as f:
            f.write(contents)

    except Exception as e:
        # Clean up partial file if exists
        if file_path.exists():
            file_path.unlink()

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save file: {str(e)}",
        ) from e

    # Create job in database
    job = Job(
        filename=safe_filename,
        original_path=str(file_path),
        source_language=source_language,
        target_language=target_language,
        voice_id=voice_id,
        context=context,
        status=JobStatus.PENDING,
        progress=0.0,
    )

    db.add(job)
    db.commit()
    db.refresh(job)

    # Trigger background processing task
    background_tasks.add_task(
        process_audio,
        job_id=job.id,
        file_path=str(file_path),
        source_lang=source_language,
        target_lang=target_language,
        voice_id=voice_id,
        context=context or "",
    )

    return job


@router.get("/jobs", response_model=list[JobResponse])
async def list_jobs(request: Request, db: Session = Depends(get_db)) -> list[Job] | HTMLResponse:
    """
    List all translation jobs.

    Args:
        request: FastAPI request object
        db: Database session

    Returns:
        List of all jobs (JSON or HTML based on Accept header)
    """
    jobs = db.query(Job).order_by(Job.created_at.desc()).all()

    # Return HTML if requested via HTMX or browser
    if "text/html" in request.headers.get("accept", "") or "hx-request" in request.headers:
        return templates.TemplateResponse("jobs_list.html", {"request": request, "jobs": jobs})

    return jobs


@router.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job(job_id: int, db: Session = Depends(get_db)) -> Job:
    """
    Get details of a specific job.

    Args:
        job_id: Job ID
        db: Database session

    Returns:
        Job details

    Raises:
        HTTPException: If job not found
    """
    job = db.query(Job).filter(Job.id == job_id).first()

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found",
        )

    return job


@router.delete("/jobs/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_job(job_id: int, db: Session = Depends(get_db)) -> None:
    """
    Delete a job and its associated files.

    Args:
        job_id: Job ID
        db: Database session

    Raises:
        HTTPException: If job not found
    """
    job = db.query(Job).filter(Job.id == job_id).first()

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found",
        )

    # Delete associated files
    try:
        # Delete original file
        if job.original_path:
            original_path = Path(job.original_path)
            if original_path.exists():
                original_path.unlink()

        # Delete output file
        if job.output_path:
            output_path = Path(job.output_path)
            if output_path.exists():
                output_path.unlink()

    except Exception as e:
        # Log error but continue with deletion
        print(f"Error deleting files for job {job_id}: {e}")

    # Delete job from database
    db.delete(job)
    db.commit()


@router.get("/jobs/{job_id}/download")
async def download_job(job_id: int, db: Session = Depends(get_db)) -> FileResponse:
    """
    Download the translated audio file.

    Args:
        job_id: Job ID
        db: Database session

    Returns:
        Audio file download

    Raises:
        HTTPException: If job not found, not completed, or file missing
    """
    job = db.query(Job).filter(Job.id == job_id).first()

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found",
        )

    if job.status != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Job is not completed yet. Current status: {job.status}",
        )

    if not job.output_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Output file not found",
        )

    output_path = Path(job.output_path)
    if not output_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Output file does not exist",
        )

    # Generate download filename
    download_filename = f"{job.filename.rsplit('.', 1)[0]}_{job.target_language}.mp3"

    return FileResponse(
        path=output_path,
        media_type="audio/mpeg",
        filename=download_filename,
        headers={
            "Content-Disposition": f'attachment; filename="{download_filename}"',
        },
    )


@router.get("/voices", response_model=list[VoiceInfo])
async def list_voices(language: str | None = None) -> list[VoiceInfo]:
    """
    List available voices for text-to-speech.

    Args:
        language: Optional language filter (e.g., 'en', 'ro', 'es')

    Returns:
        List of available voices with metadata
    """
    from app.services.tts_service import get_tts_engine

    # Get TTS engine (Piper by default)
    engine = get_tts_engine()

    # Get voices from engine
    voices = engine.list_voices(language)

    return voices


@router.post("/settings", status_code=status.HTTP_200_OK)
async def update_settings(settings_update: SettingsUpdate) -> dict[str, str]:
    """
    Update application settings (API keys, configuration).

    Args:
        settings_update: Settings to update

    Returns:
        Success message

    Note:
        In production, these should be stored securely (encrypted).
        For now, this is a placeholder endpoint.
    """
    # TODO: Implement secure storage of settings
    # For Phase 2, we'll just acknowledge the request

    updated_fields = []

    if settings_update.anthropic_api_key:
        # TODO: Store encrypted API key
        updated_fields.append("anthropic_api_key")

    if settings_update.whisper_model:
        updated_fields.append("whisper_model")

    if settings_update.tts_engine:
        updated_fields.append("tts_engine")

    return {
        "status": "success",
        "message": f"Updated settings: {', '.join(updated_fields) if updated_fields else 'none'}",
    }
