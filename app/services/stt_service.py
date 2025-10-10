"""Speech-to-Text service using Faster-Whisper."""

import asyncio
import logging
from pathlib import Path
from threading import Lock

from faster_whisper import WhisperModel

from app.config import get_settings

logger = logging.getLogger(__name__)

_stt_service_singleton: "STTService | None" = None
_stt_service_lock = Lock()


class STTService:
    """
    Speech-to-Text service using Faster-Whisper.

    Automatically detects GPU/CPU and initializes Whisper model
    with optimal compute type for the available hardware.
    """

    def __init__(self) -> None:
        """Initialize the STT service with Whisper model."""
        self.settings = get_settings()
        self.model: WhisperModel | None = None
        self._initialize_model()

    def _initialize_model(self) -> None:
        """
        Initialize Whisper model with optimal settings.

        Raises:
            RuntimeError: If model initialization fails
        """
        try:
            device = self.settings.device
            compute_type = self.settings.compute_type

            logger.info(
                f"Initializing Whisper model '{self.settings.whisper_model}' "
                f"on {device} with compute_type={compute_type}"
            )

            # Initialize Faster-Whisper model
            self.model = WhisperModel(
                model_size_or_path=self.settings.whisper_model,
                device=device,
                compute_type=compute_type,
                download_root=str(self.settings.model_dir),
            )

            logger.info("Whisper model initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Whisper model: {e}")
            raise RuntimeError(f"Failed to initialize Whisper model: {e}") from e

    async def transcribe(
        self,
        file_path: str | Path,
        language: str = "en",
        beam_size: int = 5,
        vad_filter: bool | None = None,
    ) -> str:
        """
        Transcribe audio file to text.

        Args:
            file_path: Path to audio file (MP3, WAV, etc.)
            language: Language code (e.g., 'en', 'ro', 'es')
            beam_size: Beam size for decoding (higher = more accurate, slower)
            vad_filter: Whether to use Voice Activity Detection to filter silence (defaults to settings)

        Returns:
            Transcribed text as a single string

        Raises:
            FileNotFoundError: If audio file doesn't exist
            ValueError: If file is corrupted or invalid format
            RuntimeError: If transcription fails
        """
        # Use configured VAD filter setting if not explicitly provided
        if vad_filter is None:
            vad_filter = self.settings.whisper_vad_filter

        path_obj = Path(file_path)

        # Validate file exists
        if not path_obj.exists():
            raise FileNotFoundError(f"Audio file not found: {path_obj}")

        # Validate file is not empty
        if path_obj.stat().st_size == 0:
            raise ValueError(f"Audio file is empty: {path_obj}")

        if self.model is None:
            raise RuntimeError("Whisper model not initialized")

        return await asyncio.to_thread(
            self._transcribe_sync,
            path_obj,
            language,
            beam_size,
            vad_filter,
        )

    def _transcribe_sync(
        self,
        file_path: Path,
        language: str,
        beam_size: int,
        vad_filter: bool,
    ) -> str:
        """Synchronously transcribe audio; meant to run in a worker thread."""
        try:
            model = self.model
            assert model is not None, "Whisper model must be initialized before transcription"
            logger.info(f"Starting transcription of {file_path}")

            # Debug logging for STT parameters
            if self.settings.debug:
                logger.debug("=" * 80)
                logger.debug("STT TRANSCRIPTION:")
                logger.debug(f"File: {file_path}")
                logger.debug(f"Language: {language}")
                logger.debug(f"Beam Size: {beam_size}")
                logger.debug(f"VAD Filter: {vad_filter}")
                logger.debug(f"Model: {self.settings.whisper_model}")
                logger.debug(f"Device: {self.settings.device}")
                logger.debug(f"Compute Type: {self.settings.compute_type}")
                logger.debug("=" * 80)

            # Transcribe with Faster-Whisper
            segments, info = model.transcribe(
                str(file_path),
                language=language,
                beam_size=beam_size,
                vad_filter=vad_filter,
                word_timestamps=False,
            )

            transcript_parts = [segment.text for segment in segments]
            transcript = " ".join(transcript_parts).strip()

            logger.info(
                f"Transcription completed: {len(transcript)} characters, "
                f"language={info.language} (probability={info.language_probability:.2f})"
            )

            # Debug logging for transcription result
            if self.settings.debug:
                logger.debug("=" * 80)
                logger.debug("STT RESULT:")
                logger.debug(
                    f"Detected Language: {info.language} (confidence: {info.language_probability:.2%})"
                )
                logger.debug(f"Duration: {info.duration:.2f}s")
                logger.debug(f"Transcript Length: {len(transcript)} characters")
                logger.debug(
                    f"Transcript Preview:\n{transcript[:500]}..."
                    if len(transcript) > 500
                    else f"Transcript:\n{transcript}"
                )
                logger.debug("=" * 80)

            return transcript

        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error(f"Transcription failed for {file_path}: {exc}")
            raise RuntimeError(f"Transcription failed: {exc}") from exc

    def validate_audio(self, file_path: str | Path) -> dict[str, str | int | float]:
        """
        Validate audio file and return metadata.

        Args:
            file_path: Path to audio file

        Returns:
            Dictionary with audio metadata (format, duration, size, etc.)

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is invalid or corrupted
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        if file_path.stat().st_size == 0:
            raise ValueError(f"Audio file is empty: {file_path}")

        # Basic validation - file size and extension
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        file_extension = file_path.suffix.lower()

        supported_formats = [".mp3", ".wav", ".m4a", ".ogg", ".flac", ".mp4"]
        if file_extension not in supported_formats:
            raise ValueError(
                f"Unsupported audio format: {file_extension}. "
                f"Supported formats: {', '.join(supported_formats)}"
            )

        # Check file size limit
        if file_size_mb > self.settings.max_upload_size_mb:
            raise ValueError(
                f"File too large: {file_size_mb:.2f}MB "
                f"(max: {self.settings.max_upload_size_mb}MB)"
            )

        logger.info(f"Audio file validated: {file_path} ({file_size_mb:.2f}MB)")

        return {
            "path": str(file_path),
            "size_mb": round(file_size_mb, 2),
            "format": file_extension,
        }

    def get_model_info(self) -> dict[str, str]:
        """
        Get information about the loaded Whisper model.

        Returns:
            Dictionary with model information
        """
        return {
            "model_name": self.settings.whisper_model,
            "device": self.settings.device,
            "compute_type": self.settings.compute_type,
            "model_dir": str(self.settings.model_dir),
        }


def get_stt_service() -> STTService:
    """Return a cached STT service instance, initialising it lazily."""
    global _stt_service_singleton
    if _stt_service_singleton is not None:
        return _stt_service_singleton

    with _stt_service_lock:
        if _stt_service_singleton is None:
            _stt_service_singleton = STTService()
    return _stt_service_singleton
