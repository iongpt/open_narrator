"""Text-to-Speech service with factory pattern for multiple TTS engines."""

import asyncio
import logging
from collections.abc import Callable
from pathlib import Path
from uuid import uuid4

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
        *,
        job_id: int | None = None,
    ) -> str:
        """
        Generate audio from text with progress tracking.

        Args:
            text: Text to convert to speech
            voice_id: Voice ID to use
            language: Language code
            progress_callback: Optional callback for progress updates (0.0 to 1.0)
            job_id: Optional job identifier for collision-free filenames

        Returns:
            Path to generated audio file (MP3)

        Raises:
            ValueError: If inputs are invalid
            RuntimeError: If audio generation fails
        """
        if not text:
            raise ValueError("Text cannot be empty")

        logger.info(f"Generating audio: {len(text)} chars, voice={voice_id}, lang={language}")

        loop = asyncio.get_running_loop()

        def thread_progress(value: float) -> None:
            if progress_callback is None:
                return
            loop.call_soon_threadsafe(progress_callback, value)

        if progress_callback:
            progress_callback(0.0)

        return await asyncio.to_thread(
            self._generate_audio_sync,
            text,
            voice_id,
            language,
            thread_progress if progress_callback else None,
            job_id,
        )

    def _generate_audio_sync(
        self,
        text: str,
        voice_id: str,
        language: str,
        progress_callback: Callable[[float], None] | None,
        job_id: int | None,
    ) -> str:
        """Blocking audio generation executed inside a worker thread."""
        # Debug logging for TTS parameters
        if settings.debug:
            logger.debug("=" * 80)
            logger.debug("TTS GENERATION:")
            logger.debug(f"Voice ID: {voice_id}")
            logger.debug(f"Language: {language}")
            logger.debug(f"Job ID: {job_id}")
            logger.debug(f"Engine: {self.engine.__class__.__name__}")
            logger.debug(f"Text Length: {len(text)} characters")
            logger.debug(f"Text Preview:\n{text[:500]}..." if len(text) > 500 else f"Text:\n{text}")
            logger.debug("=" * 80)

        # For better progress tracking, split text into sentences
        # This provides granular progress updates even for short/medium texts
        sentences = self._split_into_sentences(text)

        # If only 1-2 sentences, just generate directly
        if len(sentences) <= 2:
            logger.info(f"Generating audio for short text ({len(sentences)} sentence(s))")
            try:
                output_path = self.engine.generate_audio(text, voice_id, language)

                # Debug logging for TTS result
                if settings.debug:
                    logger.debug("=" * 80)
                    logger.debug("TTS RESULT:")
                    logger.debug(f"Output Path: {output_path}")
                    logger.debug(f"File Size: {Path(output_path).stat().st_size / 1024:.2f} KB")
                    logger.debug("=" * 80)

                if progress_callback:
                    progress_callback(1.0)
                return output_path
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.error(f"Audio generation failed: {exc}")
                raise RuntimeError(f"Failed to generate audio: {exc}") from exc

        # For multiple sentences, generate with progress tracking
        logger.info(f"Generating audio with progress tracking ({len(sentences)} sentences)")
        return self._generate_with_sentence_progress(
            sentences,
            voice_id,
            language,
            progress_callback,
            job_id,
        )

    def _generate_long_audio_sync(
        self,
        text: str,
        voice_id: str,
        language: str,
        progress_callback: Callable[[float], None] | None,
        job_id: int | None,
    ) -> str:
        """
        Generate audio for long text by splitting into chunks.

        Args:
            text: Long text to convert
            voice_id: Voice ID to use
            language: Language code
            progress_callback: Optional progress callback
            job_id: Optional job identifier for unique filenames

        Returns:
            Path to concatenated audio file

        Raises:
            RuntimeError: If audio generation or concatenation fails
        """
        chunks = self._split_text(text, max_length=5000)
        logger.info(f"Split text into {len(chunks)} chunks")

        from pydub import AudioSegment

        audio_paths: list[str] = []
        combined_audio = AudioSegment.empty()

        for i, chunk in enumerate(chunks):
            logger.info(f"Generating chunk {i + 1}/{len(chunks)}")
            chunk_path = self.engine.generate_audio(chunk, voice_id, language)
            audio_paths.append(chunk_path)

            chunk_audio = AudioSegment.from_mp3(chunk_path)
            combined_audio += chunk_audio

            if progress_callback:
                progress = (i + 1) / len(chunks)
                progress_callback(progress)

        unique_suffix = uuid4().hex[:8]
        job_prefix = f"job{job_id}_" if job_id is not None else ""
        output_path = settings.output_dir / f"{job_prefix}{voice_id}_{unique_suffix}.mp3"
        combined_audio.export(
            str(output_path),
            format="mp3",
            bitrate="128k",
            parameters=["-ar", "22050"],
        )

        logger.info(f"Generated combined audio: {output_path}")

        for path in audio_paths:
            try:
                Path(path).unlink()
            except Exception as exc:  # pragma: no cover - cleanup best-effort
                logger.warning(f"Failed to delete chunk file {path}: {exc}")

        if progress_callback:
            progress_callback(1.0)

        return str(output_path)

    def _split_into_sentences(self, text: str) -> list[str]:
        """
        Split text into individual sentences for progress tracking.

        Args:
            text: Text to split into sentences

        Returns:
            List of sentences
        """
        import re

        # Split on sentence boundaries (., !, ?, and newlines)
        # Keep the punctuation with the sentence
        sentences = re.split(r"(?<=[.!?])\s+|\n+", text)

        # Filter out empty sentences and strip whitespace
        sentences = [s.strip() for s in sentences if s.strip()]

        return sentences

    def _generate_with_sentence_progress(
        self,
        sentences: list[str],
        voice_id: str,
        language: str,
        progress_callback: Callable[[float], None] | None,
        job_id: int | None,
    ) -> str:
        """
        Generate audio for multiple sentences with progress tracking.

        Args:
            sentences: List of sentences to generate audio for
            voice_id: Voice ID to use
            language: Language code
            progress_callback: Optional progress callback
            job_id: Optional job identifier for unique filenames

        Returns:
            Path to combined audio file

        Raises:
            RuntimeError: If audio generation or concatenation fails
        """
        from pydub import AudioSegment

        total_sentences = len(sentences)
        logger.info(f"Generating audio for {total_sentences} sentences")

        audio_paths: list[str] = []
        combined_audio = AudioSegment.empty()

        for i, sentence in enumerate(sentences):
            logger.info(f"Generating sentence {i + 1}/{total_sentences}")

            # Generate audio for this sentence
            sentence_path = self.engine.generate_audio(sentence, voice_id, language)
            audio_paths.append(sentence_path)

            # Add to combined audio
            sentence_audio = AudioSegment.from_mp3(sentence_path)
            combined_audio += sentence_audio

            # Report progress
            if progress_callback:
                progress = (i + 1) / total_sentences
                progress_callback(progress)

        # Save combined audio
        from uuid import uuid4

        unique_suffix = uuid4().hex[:8]
        job_prefix = f"job{job_id}_" if job_id is not None else ""
        output_path = settings.output_dir / f"{job_prefix}{voice_id}_{unique_suffix}.mp3"

        combined_audio.export(
            str(output_path),
            format="mp3",
            bitrate="128k",
            parameters=["-ar", "22050"],
        )

        logger.info(f"Generated combined audio: {output_path}")

        # Clean up individual sentence files
        for path in audio_paths:
            try:
                Path(path).unlink()
            except Exception as exc:  # pragma: no cover - cleanup best-effort
                logger.warning(f"Failed to delete sentence file {path}: {exc}")

        if progress_callback:
            progress_callback(1.0)

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
