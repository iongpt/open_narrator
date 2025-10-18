"""Base abstraction for Text-to-Speech engines."""

from abc import ABC, abstractmethod

from app.schemas import VoiceInfo


class BaseTTSEngine(ABC):
    """
    Abstract base class for TTS engines.

    This class defines the interface that all TTS engine implementations
    must follow, allowing easy swapping between different TTS providers
    (Piper, XTTS, Coqui, etc.).
    """

    @abstractmethod
    def generate_audio(
        self,
        text: str,
        voice_id: str,
        language: str,
        *,
        length_scale: float | None = None,
        noise_scale: float | None = None,
    ) -> str:
        """
        Generate audio from text using the specified voice.

        Args:
            text: The text to convert to speech
            voice_id: The ID of the voice to use
            language: Language code (e.g., 'en', 'ro', 'es')
            length_scale: Optional tempo multiplier (lower is faster, higher slower)
            noise_scale: Optional prosody randomness control

        Returns:
            Path to the generated audio file (MP3 format)

        Raises:
            ValueError: If voice_id is invalid or language not supported
            RuntimeError: If audio generation fails
        """
        pass

    @abstractmethod
    def list_voices(self, language: str | None = None) -> list[VoiceInfo]:
        """
        List available voices, optionally filtered by language.

        Args:
            language: Optional language code to filter voices (e.g., 'en', 'ro')

        Returns:
            List of VoiceInfo objects containing voice metadata

        Raises:
            RuntimeError: If voice listing fails
        """
        pass

    @abstractmethod
    def get_voice_info(self, voice_id: str) -> VoiceInfo:
        """
        Get detailed information about a specific voice.

        Args:
            voice_id: The ID of the voice to get info for

        Returns:
            VoiceInfo object with voice metadata

        Raises:
            ValueError: If voice_id is not found
        """
        pass

    @abstractmethod
    def is_voice_available(self, voice_id: str) -> bool:
        """
        Check if a voice model is available locally.

        Args:
            voice_id: The ID of the voice to check

        Returns:
            True if voice model is downloaded and ready, False otherwise
        """
        pass

    @abstractmethod
    def download_voice(self, voice_id: str) -> None:
        """
        Download a voice model if not already available.

        Args:
            voice_id: The ID of the voice to download

        Raises:
            ValueError: If voice_id is invalid
            RuntimeError: If download fails
        """
        pass
