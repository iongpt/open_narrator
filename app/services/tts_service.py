"""Text-to-Speech service with factory pattern for multiple TTS engines."""

import logging
from collections.abc import Callable
from pathlib import Path

from app.config import get_settings
from app.schemas import VoiceInfo
from app.tts_engines.base import BaseTTSEngine
from app.tts_engines.piper import PiperEngine

logger = logging.getLogger(__name__)
settings = get_settings()


def get_tts_engine(engine_name: str | None = None) -> BaseTTSEngine:
    """
    Factory function to get TTS engine instance.

    Args:
        engine_name: Name of the TTS engine ('piper', etc.).
                    If None, uses default from settings.

    Returns:
        TTS engine instance

    Raises:
        ValueError: If engine_name is unknown
    """
    if engine_name is None:
        engine_name = settings.tts_engine

    engines = {
        "piper": PiperEngine,
        # Easy to add more engines:
        # "xtts": XTTSEngine,
        # "coqui": CoquiEngine,
    }

    if engine_name not in engines:
        raise ValueError(
            f"Unknown TTS engine: {engine_name}. " f"Available engines: {', '.join(engines.keys())}"
        )

    return engines[engine_name]()


class TTSService:
    """
    Main TTS service that wraps TTS engines and provides additional functionality.

    This service handles:
    - Long text processing (splitting and concatenation)
    - Progress tracking
    - Audio post-processing
    - Error handling
    """

    def __init__(self, engine: BaseTTSEngine | None = None) -> None:
        """
        Initialize TTS service.

        Args:
            engine: TTS engine to use. If None, uses default from settings.
        """
        self.engine = engine or get_tts_engine()

    async def generate_audio(
        self,
        text: str,
        voice_id: str,
        language: str,
        progress_callback: Callable[[float], None] | None = None,
    ) -> str:
        """
        Generate audio from text with progress tracking.

        Args:
            text: Text to convert to speech
            voice_id: Voice ID to use
            language: Language code
            progress_callback: Optional callback for progress updates (0.0 to 1.0)

        Returns:
            Path to generated audio file (MP3)

        Raises:
            ValueError: If inputs are invalid
            RuntimeError: If audio generation fails
        """
        if not text:
            raise ValueError("Text cannot be empty")

        logger.info(f"Generating audio: {len(text)} chars, voice={voice_id}, lang={language}")

        # Update progress: starting
        if progress_callback:
            progress_callback(0.0)

        # Check if text is too long (>10000 chars, split it)
        if len(text) > 10000:
            logger.info("Text is long, splitting into chunks")
            return await self._generate_long_audio(text, voice_id, language, progress_callback)

        # Generate audio in one go
        try:
            output_path = self.engine.generate_audio(text, voice_id, language)

            # Update progress: complete
            if progress_callback:
                progress_callback(1.0)

            return output_path

        except Exception as e:
            logger.error(f"Audio generation failed: {e}")
            raise RuntimeError(f"Failed to generate audio: {str(e)}") from e

    async def _generate_long_audio(
        self,
        text: str,
        voice_id: str,
        language: str,
        progress_callback: Callable[[float], None] | None = None,
    ) -> str:
        """
        Generate audio for long text by splitting into chunks.

        Args:
            text: Long text to convert
            voice_id: Voice ID to use
            language: Language code
            progress_callback: Optional progress callback

        Returns:
            Path to concatenated audio file

        Raises:
            RuntimeError: If audio generation or concatenation fails
        """
        # Split text into chunks by sentences/paragraphs
        chunks = self._split_text(text, max_length=5000)
        logger.info(f"Split text into {len(chunks)} chunks")

        # Import here to avoid circular dependency
        from pydub import AudioSegment

        # Generate audio for each chunk
        audio_paths = []
        combined_audio = AudioSegment.empty()

        for i, chunk in enumerate(chunks):
            logger.info(f"Generating chunk {i+1}/{len(chunks)}")

            # Generate audio for chunk
            chunk_path = self.engine.generate_audio(chunk, voice_id, language)
            audio_paths.append(chunk_path)

            # Load and append to combined audio
            chunk_audio = AudioSegment.from_mp3(chunk_path)
            combined_audio += chunk_audio

            # Update progress
            if progress_callback:
                progress = (i + 1) / len(chunks)
                progress_callback(progress)

        # Export combined audio
        output_path = settings.output_dir / f"combined_{voice_id}.mp3"
        combined_audio.export(
            str(output_path), format="mp3", bitrate="128k", parameters=["-ar", "22050"]
        )

        logger.info(f"Generated combined audio: {output_path}")

        # Clean up individual chunk files
        for path in audio_paths:
            try:
                Path(path).unlink()
            except Exception as e:
                logger.warning(f"Failed to delete chunk file {path}: {e}")

        return str(output_path)

    def _split_text(self, text: str, max_length: int = 5000) -> list[str]:
        """
        Split long text into chunks at sentence boundaries.

        Args:
            text: Text to split
            max_length: Maximum characters per chunk

        Returns:
            List of text chunks
        """
        # Split by sentence endings
        import re

        # Split on sentence boundaries (., !, ?)
        sentences = re.split(r"([.!?]+\s+)", text)

        chunks = []
        current_chunk = ""

        for i in range(0, len(sentences), 2):
            sentence = sentences[i]

            # Add punctuation if exists
            if i + 1 < len(sentences):
                sentence += sentences[i + 1]

            # Check if adding sentence exceeds max length
            if len(current_chunk) + len(sentence) > max_length and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += sentence

        # Add remaining text
        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def list_voices(self, language: str | None = None) -> list[VoiceInfo]:
        """
        List available voices from the TTS engine.

        Args:
            language: Optional language filter

        Returns:
            List of available voices
        """
        return self.engine.list_voices(language)

    def get_voice_info(self, voice_id: str) -> VoiceInfo:
        """
        Get information about a specific voice.

        Args:
            voice_id: Voice ID to get info for

        Returns:
            Voice information

        Raises:
            ValueError: If voice_id is not found
        """
        return self.engine.get_voice_info(voice_id)

    def is_voice_available(self, voice_id: str) -> bool:
        """
        Check if a voice is available locally.

        Args:
            voice_id: Voice ID to check

        Returns:
            True if voice is available, False otherwise
        """
        return self.engine.is_voice_available(voice_id)

    def download_voice(self, voice_id: str) -> None:
        """
        Download a voice model if not available.

        Args:
            voice_id: Voice ID to download

        Raises:
            ValueError: If voice_id is invalid
            RuntimeError: If download fails
        """
        self.engine.download_voice(voice_id)
