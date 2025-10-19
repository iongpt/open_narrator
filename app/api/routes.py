"""API routes for OpenNarrator."""

import math
import os
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
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
ALLOWED_AUDIO_EXTENSIONS = {
    ".mp3",
    ".wav",
    ".m4a",
    ".ogg",
    ".flac",
    ".mp4",
}

ALLOWED_AUDIO_TYPES = {
    "audio/mpeg",
    "audio/mp3",
    "audio/x-mp3",
    "audio/mpeg3",
    "audio/x-mpeg-3",
    "audio/wav",
    "audio/x-wav",
    "audio/wave",
    "audio/mp4",
    "audio/m4a",
    "audio/x-m4a",
    "audio/ogg",
    "audio/flac",
}

ALLOWED_TEXT_TYPES = {
    "text/plain",
    "text/markdown",
    "text/html",
    "application/pdf",
    "application/epub+zip",
    "application/x-mobipocket-ebook",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # DOCX
    "application/msword",  # DOC
    "application/rtf",
    "text/rtf",
    "application/vnd.oasis.opendocument.text",  # ODT
}

ALLOWED_TEXT_EXTENSIONS = {
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

MAX_FILE_SIZE = settings.max_upload_size_mb * 1024 * 1024  # Convert to bytes


def validate_file(file: UploadFile) -> str:
    """
    Validate uploaded file (audio or text).

    Args:
        file: The uploaded file to validate

    Returns:
        File type: 'audio' or 'text'

    Raises:
        HTTPException: If file is invalid
    """
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename is required",
        )

    filename_lower = file.filename.lower()
    file_extension = Path(filename_lower).suffix

    # Check if it's an audio file (by extension OR content-type)
    if file_extension in ALLOWED_AUDIO_EXTENSIONS or file.content_type in ALLOWED_AUDIO_TYPES:
        return "audio"

    # Check if it's a text file (by extension OR content-type)
    if file_extension in ALLOWED_TEXT_EXTENSIONS or file.content_type in ALLOWED_TEXT_TYPES:
        return "text"

    # Invalid file type - provide helpful error message
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=(
            "Invalid file type. Allowed formats:\n"
            f"Audio: {', '.join(sorted(ALLOWED_AUDIO_EXTENSIONS))}\n"
            f"Text: {', '.join(sorted(ALLOWED_TEXT_EXTENSIONS))}"
        ),
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
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    source_language: str = Form("en"),
    target_language: str = Form(...),
    voice_id: str = Form(...),
    context: str | None = Form(None),
    skip_translation: bool = Form(False),
    length_scale: float | None = Form(1.0),
    noise_scale: float | None = Form(1.0),
    noise_w_scale: float | None = Form(1.0),
    db: Session = Depends(get_db),
) -> Job:
    """
    Upload an audio or text file for translation.

    Args:
        background_tasks: FastAPI background tasks manager
        file: The audio or text file to upload
        source_language: Source language code (default: en)
        target_language: Target language code
        voice_id: Voice ID for TTS
        context: Optional context for translation
        skip_translation: Skip translation (content already in target language)
        db: Database session

    Returns:
        Created job details

    Raises:
        HTTPException: If file validation fails or upload error occurs
    """
    # Validate file and get type
    file_type = validate_file(file)

    # Validate skip_translation: only text files allowed
    if skip_translation and file_type == "audio":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Audio files cannot be used with skip_translation. Only text files are supported when content is already in target language.",
        )

    # Sanitize filename
    safe_filename = sanitize_filename(file.filename or "upload.mp3")

    # Create unique filename to avoid collisions
    unique_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex}_{safe_filename}"

    # Save file
    file_path = settings.upload_dir / unique_filename

    try:
        total_bytes = 0
        chunk_size = 1024 * 1024  # 1MB chunks to bound memory usage

        with open(file_path, "wb") as f:
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break

                total_bytes += len(chunk)
                if total_bytes > MAX_FILE_SIZE:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=(f"File too large. Maximum size: {settings.max_upload_size_mb}MB"),
                    )

                f.write(chunk)

    except Exception as e:
        # Clean up partial file if exists
        if file_path.exists():
            file_path.unlink()

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save file: {str(e)}",
        ) from e

    # Normalize scale sliders: treat 1.0 as "use model default"
    if length_scale is not None and math.isclose(length_scale, 1.0, rel_tol=1e-3):
        length_scale = None
    if noise_scale is not None and math.isclose(noise_scale, 1.0, rel_tol=1e-3):
        noise_scale = None
    if noise_w_scale is not None and math.isclose(noise_w_scale, 1.0, rel_tol=1e-3):
        noise_w_scale = None

    # Create job in database
    job = Job(
        filename=safe_filename,
        original_path=str(file_path),
        source_language=source_language,
        target_language=target_language,
        voice_id=voice_id,
        context=context,
        skip_translation=skip_translation,
        length_scale=length_scale,
        noise_scale=noise_scale,
        noise_w_scale=noise_w_scale,
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
        file_type=file_type,  # Pass file type to pipeline
        skip_translation=skip_translation,  # Pass skip_translation flag to pipeline
        length_scale=length_scale,
        noise_scale=noise_scale,
        noise_w_scale=noise_w_scale,
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


@router.get("/jobs/{job_id}/stream")
async def stream_job_audio(job_id: int, db: Session = Depends(get_db)) -> FileResponse:
    """Stream the generated audio file for in-browser playback."""

    job = db.query(Job).filter(Job.id == job_id).first()

    if not job or job.status != JobStatus.COMPLETED or not job.output_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Audio for job {job_id} not found",
        )

    output_path = Path(job.output_path)

    if not output_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Audio file not found for job {job_id}",
        )

    return FileResponse(
        path=output_path,
        media_type="audio/mpeg",
    )


@router.get("/jobs/{job_id}/player", response_class=HTMLResponse)
async def job_audio_player(
    request: Request, job_id: int, db: Session = Depends(get_db)
) -> HTMLResponse:
    """Render a simple audio player page for a completed job."""

    job = db.query(Job).filter(Job.id == job_id).first()

    if not job or job.status != JobStatus.COMPLETED or not job.output_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Audio for job {job_id} not found",
        )

    return templates.TemplateResponse(
        "audio_player.html",
        {
            "request": request,
            "job": job,
            "stream_url": f"/api/jobs/{job_id}/stream",
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
    from app.services.tts_service import TTSService

    service = TTSService()
    return service.list_voices(language)


@router.get("/voices/{voice_id}/preview")
async def preview_voice(
    voice_id: str,
    length_scale: float | None = Query(None, gt=0.1, lt=5.0),
    noise_scale: float | None = Query(None, ge=0.0, lt=5.0),
    noise_w_scale: float | None = Query(None, ge=0.0, lt=5.0),
) -> FileResponse:
    """
    Generate and serve a preview audio sample for a voice.

    Args:
        voice_id: The ID of the voice to preview

    Returns:
        Audio file (MP3) with sample text

    Raises:
        HTTPException: If voice_id is invalid or preview generation fails
    """
    import logging
    import shutil

    from app.services.tts_service import TTSService

    logger = logging.getLogger(__name__)

    try:
        # Resolve voice metadata and appropriate engine
        tts_service = TTSService()
        voice_info = tts_service.get_voice_info(voice_id)

        # Sample text in different languages (pre-prepared)
        sample_texts = {
            "en": "Welcome! This is a preview of how this voice sounds for your audiobook translation.",
            "ro": "Bună ziua! Aceasta este o previzualizare a acestei voci pentru traducerea cărții tale audio.",
            "es": "¡Bienvenido! Esta es una muestra de cómo suena esta voz para tu audiolibro traducido.",
            "fr": "Bienvenue! Ceci est un aperçu de cette voix pour votre livre audio traduit.",
            "de": "Willkommen! Dies ist eine Hörprobe dieser Stimme für Ihr übersetztes Hörbuch.",
            "it": "Benvenuto! Questa è un'anteprima di come suona questa voce per il tuo audiolibro tradotto.",
            "pt": "Bem-vindo! Esta é uma prévia de como esta voz soa para seu audiolivro traduzido.",
            "nl": "Welkom! Dit is een voorproefje van hoe deze stem klinkt voor uw vertaalde audioboek.",
            "pl": "Witaj! To jest podgląd tego głosu dla Twojego przetłumaczonego audiobooka.",
            "ru": "Добро пожаловать! Это образец звучания этого голоса для вашей переведённой аудиокниги.",
            "uk": "Ласкаво просимо! Це зразок звучання цього голосу для вашої перекладеної аудіокниги.",
            "ja": "ようこそ！これはあなたの翻訳されたオーディオブックのための音声サンプルです。",
            "zh": "欢迎！这是您翻译的有声读物的语音示例。",
            "ko": "환영합니다! 번역된 오디오북을 위한 음성 샘플입니다.",
            "ar": "مرحباً! هذه عينة صوتية لكتابك الصوتي المترجم.",
            "hi": "स्वागत है! यह आपकी अनुवादित ऑडियोबुक के लिए आवाज़ का नमूना है।",
            "tr": "Hoş geldiniz! Bu, çevrilmiş sesli kitabınız için ses örneğidir.",
        }

        # Treat near-default slider values as None to use model config defaults
        if length_scale is not None and math.isclose(length_scale, 1.0, rel_tol=1e-3):
            length_scale = None
        if noise_scale is not None and math.isclose(noise_scale, 1.0, rel_tol=1e-3):
            noise_scale = None
        if noise_w_scale is not None and math.isclose(noise_w_scale, 1.0, rel_tol=1e-3):
            noise_w_scale = None

        # Get sample text for voice language
        sample_text = sample_texts.get(voice_info.language, sample_texts["en"])

        # Check if sample already exists (CACHE)
        sample_dir = settings.static_dir / "voice_samples"
        sample_dir.mkdir(parents=True, exist_ok=True)
        ls_tag = "default" if length_scale is None else f"{length_scale:.2f}"
        ns_tag = "default" if noise_scale is None else f"{noise_scale:.2f}"
        nws_tag = "default" if noise_w_scale is None else f"{noise_w_scale:.2f}"
        safe_voice_id = voice_id.replace(":", "_")
        sample_filename = f"{safe_voice_id}_ls{ls_tag}_ns{ns_tag}_nws{nws_tag}.mp3"
        sample_path = sample_dir / sample_filename

        # Return cached version if it exists
        if sample_path.exists():
            logger.info(f"Serving cached voice preview for {voice_id}")
            return FileResponse(
                path=sample_path,
                media_type="audio/mpeg",
                filename=sample_filename,
            )

        # Generate sample if it doesn't exist
        logger.info(f"Generating new voice preview for {voice_id}")
        audio_path = await tts_service.generate_audio(
            text=sample_text,
            voice_id=voice_id,
            language=voice_info.language,
            length_scale=length_scale,
            noise_scale=noise_scale,
            noise_w_scale=noise_w_scale,
        )

        # Copy (not move) to samples directory
        try:
            shutil.copy2(audio_path, sample_path)
            logger.info(f"Copied voice preview to {sample_path}")

            # Clean up original file if it's in temp directory
            audio_path_obj = Path(audio_path)
            if audio_path_obj.exists() and "output" in str(audio_path_obj):
                audio_path_obj.unlink()
                logger.debug(f"Deleted temporary file: {audio_path}")

        except Exception as copy_error:
            logger.error(f"Failed to copy preview file: {copy_error}")
            # If copy fails, just use the generated file directly
            sample_path = Path(audio_path)

        return FileResponse(
            path=sample_path,
            media_type="audio/mpeg",
            filename=sample_filename,
        )

    except ValueError as e:
        logger.error(f"Invalid voice ID {voice_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        ) from e
    except Exception as e:
        logger.error(f"Failed to generate voice preview for {voice_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate voice preview: {str(e)}",
        ) from e


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
