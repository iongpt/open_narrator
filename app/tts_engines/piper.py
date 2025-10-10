"""Piper TTS engine implementation."""

import logging
import tempfile
from pathlib import Path

import httpx
from piper import PiperVoice
from pydub import AudioSegment

from app.config import get_settings
from app.schemas import VoiceInfo
from app.tts_engines.base import BaseTTSEngine

logger = logging.getLogger(__name__)
settings = get_settings()

# Piper voices repository on Hugging Face
PIPER_VOICES_BASE_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main"

# Piper voices metadata (subset of high-quality voices)
# Format: voice_id -> (name, language, gender, quality, hf_path)
PIPER_VOICES_CATALOG = {
    # English
    "en_US-lessac-medium": (
        "Lessac (US English)",
        "en",
        "male",
        "medium",
        "en/en_US/lessac/medium/en_US-lessac-medium",
    ),
    "en_US-amy-medium": (
        "Amy (US English)",
        "en",
        "female",
        "medium",
        "en/en_US/amy/medium/en_US-amy-medium",
    ),
    "en_GB-alba-medium": (
        "Alba (British English)",
        "en",
        "female",
        "medium",
        "en/en_GB/alba/medium/en_GB-alba-medium",
    ),
    "en_US-libritts-high": (
        "LibriTTS (US English)",
        "en",
        "neutral",
        "high",
        "en/en_US/libritts/high/en_US-libritts-high",
    ),
    # Romanian
    "ro_RO-mihai-medium": (
        "Mihai (Romanian)",
        "ro",
        "male",
        "medium",
        "ro/ro_RO/mihai/medium/ro_RO-mihai-medium",
    ),
    # Spanish
    "es_ES-mls_9972-low": (
        "MLS 9972 (Spanish)",
        "es",
        "female",
        "low",
        "es/es_ES/mls_9972/low/es_ES-mls_9972-low",
    ),
    "es_ES-carlfm-x_low": (
        "Carlfm (Spanish)",
        "es",
        "male",
        "x_low",
        "es/es_ES/carlfm/x_low/es_ES-carlfm-x_low",
    ),
    # French
    "fr_FR-siwis-medium": (
        "Siwis (French)",
        "fr",
        "female",
        "medium",
        "fr/fr_FR/siwis/medium/fr_FR-siwis-medium",
    ),
    "fr_FR-mls_1840-low": (
        "MLS 1840 (French)",
        "fr",
        "male",
        "low",
        "fr/fr_FR/mls_1840/low/fr_FR-mls_1840-low",
    ),
    # German
    "de_DE-thorsten-medium": (
        "Thorsten (German)",
        "de",
        "male",
        "medium",
        "de/de_DE/thorsten/medium/de_DE-thorsten-medium",
    ),
    "de_DE-eva_k-x_low": (
        "Eva K (German)",
        "de",
        "female",
        "x_low",
        "de/de_DE/eva_k/x_low/de_DE-eva_k-x_low",
    ),
    # Italian
    "it_IT-riccardo-x_low": (
        "Riccardo (Italian)",
        "it",
        "male",
        "x_low",
        "it/it_IT/riccardo/x_low/it_IT-riccardo-x_low",
    ),
    # Portuguese
    "pt_BR-faber-medium": (
        "Faber (Brazilian Portuguese)",
        "pt",
        "male",
        "medium",
        "pt/pt_BR/faber/medium/pt_BR-faber-medium",
    ),
    # Dutch
    "nl_NL-mls_5809-low": (
        "MLS 5809 (Dutch)",
        "nl",
        "female",
        "low",
        "nl/nl_NL/mls_5809/low/nl_NL-mls_5809-low",
    ),
    # Polish
    "pl_PL-mls_6892-low": (
        "MLS 6892 (Polish)",
        "pl",
        "female",
        "low",
        "pl/pl_PL/mls_6892/low/pl_PL-mls_6892-low",
    ),
    # Russian
    "ru_RU-irinia-medium": (
        "Irina (Russian)",
        "ru",
        "female",
        "medium",
        "ru/ru_RU/irinia/medium/ru_RU-irinia-medium",
    ),
    # Ukrainian
    "uk_UA-lada-x_low": (
        "Lada (Ukrainian)",
        "uk",
        "female",
        "x_low",
        "uk/uk_UA/lada/x_low/uk_UA-lada-x_low",
    ),
    # Japanese
    "ja_JP-teikyoku-medium": (
        "Teikyoku (Japanese)",
        "ja",
        "neutral",
        "medium",
        "ja/ja_JP/teikyoku/medium/ja_JP-teikyoku-medium",
    ),
    # Chinese (Mandarin)
    "zh_CN-huayan-medium": (
        "Huayan (Mandarin Chinese)",
        "zh",
        "female",
        "medium",
        "zh/zh_CN/huayan/medium/zh_CN-huayan-medium",
    ),
    # Arabic
    "ar_JO-kareem-medium": (
        "Kareem (Arabic)",
        "ar",
        "male",
        "medium",
        "ar/ar_JO/kareem/medium/ar_JO-kareem-medium",
    ),
    # Turkish
    "tr_TR-dfki-medium": (
        "DFKI (Turkish)",
        "tr",
        "male",
        "medium",
        "tr/tr_TR/dfki/medium/tr_TR-dfki-medium",
    ),
    # Hindi
    "hi_IN-medium": (
        "Medium (Hindi)",
        "hi",
        "neutral",
        "medium",
        "hi/hi_IN/medium/hi_IN-medium",
    ),
    # Korean
    "ko_KR-kss-medium": (
        "KSS (Korean)",
        "ko",
        "female",
        "medium",
        "ko/ko_KR/kss/medium/ko_KR-kss-medium",
    ),
}


class PiperEngine(BaseTTSEngine):
    """
    Piper TTS engine implementation using ONNX runtime.

    Piper is a fast, local text-to-speech system that uses ONNX models.
    It supports 30+ languages with high-quality neural voices.
    """

    def __init__(self) -> None:
        """Initialize Piper engine with configuration."""
        self.model_dir = settings.model_dir / "piper"
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Cache for loaded voice models
        self._voice_cache: dict[str, PiperVoice] = {}

    def _get_model_path(self, voice_id: str) -> Path:
        """
        Get the local path for a voice model.

        Args:
            voice_id: Voice identifier

        Returns:
            Path to the .onnx model file
        """
        return self.model_dir / f"{voice_id}.onnx"

    def _get_config_path(self, voice_id: str) -> Path:
        """
        Get the local path for a voice config.

        Args:
            voice_id: Voice identifier

        Returns:
            Path to the .onnx.json config file
        """
        return self.model_dir / f"{voice_id}.onnx.json"

    def is_voice_available(self, voice_id: str) -> bool:
        """
        Check if a voice model is available locally.

        Args:
            voice_id: The ID of the voice to check

        Returns:
            True if both model and config files exist, False otherwise
        """
        if voice_id not in PIPER_VOICES_CATALOG:
            return False

        model_path = self._get_model_path(voice_id)
        config_path = self._get_config_path(voice_id)

        return model_path.exists() and config_path.exists()

    def download_voice(self, voice_id: str) -> None:
        """
        Download a voice model from Hugging Face if not already available.

        Args:
            voice_id: The ID of the voice to download

        Raises:
            ValueError: If voice_id is invalid
            RuntimeError: If download fails
        """
        if voice_id not in PIPER_VOICES_CATALOG:
            raise ValueError(
                f"Unknown voice ID: {voice_id}. "
                f"Available voices: {', '.join(PIPER_VOICES_CATALOG.keys())}"
            )

        # Check if already downloaded
        if self.is_voice_available(voice_id):
            logger.info(f"Voice {voice_id} already available locally")
            return

        logger.info(f"Downloading voice model: {voice_id}")

        # Get HuggingFace path
        hf_path = PIPER_VOICES_CATALOG[voice_id][4]

        # Download model and config files
        model_url = f"{PIPER_VOICES_BASE_URL}/{hf_path}.onnx"
        config_url = f"{PIPER_VOICES_BASE_URL}/{hf_path}.onnx.json"

        model_path = self._get_model_path(voice_id)
        config_path = self._get_config_path(voice_id)

        try:
            # Download model file
            logger.info(f"Downloading model from {model_url}")
            with httpx.stream("GET", model_url, follow_redirects=True, timeout=300.0) as response:
                response.raise_for_status()
                with open(model_path, "wb") as f:
                    for chunk in response.iter_bytes(chunk_size=8192):
                        f.write(chunk)

            # Download config file
            logger.info(f"Downloading config from {config_url}")
            with httpx.stream("GET", config_url, follow_redirects=True, timeout=60.0) as response:
                response.raise_for_status()
                with open(config_path, "wb") as f:
                    for chunk in response.iter_bytes(chunk_size=8192):
                        f.write(chunk)

            logger.info(f"Successfully downloaded voice: {voice_id}")

        except Exception as e:
            # Clean up partial downloads
            if model_path.exists():
                model_path.unlink()
            if config_path.exists():
                config_path.unlink()

            raise RuntimeError(f"Failed to download voice {voice_id}: {str(e)}") from e

    def _load_voice(self, voice_id: str) -> PiperVoice:
        """
        Load a voice model into memory.

        Args:
            voice_id: Voice identifier

        Returns:
            Loaded PiperVoice instance

        Raises:
            ValueError: If voice is not available
            RuntimeError: If model loading fails
        """
        # Check cache
        if voice_id in self._voice_cache:
            return self._voice_cache[voice_id]

        # Ensure voice is downloaded
        if not self.is_voice_available(voice_id):
            self.download_voice(voice_id)

        # Load voice model
        try:
            model_path = self._get_model_path(voice_id)
            voice = PiperVoice.load(str(model_path))
            self._voice_cache[voice_id] = voice
            logger.info(f"Loaded voice model: {voice_id}")
            return voice

        except Exception as e:
            raise RuntimeError(f"Failed to load voice {voice_id}: {str(e)}") from e

    def generate_audio(self, text: str, voice_id: str, language: str) -> str:
        """
        Generate audio from text using the specified voice.

        Args:
            text: The text to convert to speech
            voice_id: The ID of the voice to use
            language: Language code (e.g., 'en', 'ro', 'es')

        Returns:
            Path to the generated audio file (MP3 format)

        Raises:
            ValueError: If voice_id is invalid or language not supported
            RuntimeError: If audio generation fails
        """
        if not text:
            raise ValueError("Text cannot be empty")

        if voice_id not in PIPER_VOICES_CATALOG:
            raise ValueError(
                f"Unknown voice ID: {voice_id}. "
                f"Available voices: {', '.join(PIPER_VOICES_CATALOG.keys())}"
            )

        # Verify language matches
        voice_lang = PIPER_VOICES_CATALOG[voice_id][1]
        if not language.startswith(voice_lang):
            logger.warning(f"Language mismatch: requested {language}, voice is {voice_lang}")

        logger.info(f"Generating audio with voice {voice_id} for {len(text)} characters")

        try:
            # Load voice model
            voice = self._load_voice(voice_id)

            # Create temporary WAV file (Piper outputs WAV)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_file:
                wav_path = Path(wav_file.name)

            # Generate audio
            with open(wav_path, "wb") as f:
                voice.synthesize(text, f)

            logger.info(f"Generated WAV audio: {wav_path}")

            # Convert WAV to MP3
            mp3_path = settings.output_dir / f"{wav_path.stem}.mp3"
            self._convert_to_mp3(wav_path, mp3_path)

            # Clean up WAV file
            wav_path.unlink()

            logger.info(f"Generated MP3 audio: {mp3_path}")
            return str(mp3_path)

        except Exception as e:
            raise RuntimeError(f"Failed to generate audio: {str(e)}") from e

    def _convert_to_mp3(self, wav_path: Path, mp3_path: Path) -> None:
        """
        Convert WAV audio to MP3 format.

        Args:
            wav_path: Path to input WAV file
            mp3_path: Path to output MP3 file

        Raises:
            RuntimeError: If conversion fails
        """
        try:
            # Load WAV with pydub
            audio = AudioSegment.from_wav(str(wav_path))

            # Export as MP3 (128 kbps)
            audio.export(
                str(mp3_path),
                format="mp3",
                bitrate="128k",
                parameters=["-ar", "22050"],  # 22050 Hz sample rate (Piper default)
            )

            logger.info(f"Converted WAV to MP3: {mp3_path}")

        except Exception as e:
            raise RuntimeError(f"Failed to convert audio to MP3: {str(e)}") from e

    def list_voices(self, language: str | None = None) -> list[VoiceInfo]:
        """
        List available voices, optionally filtered by language.

        Args:
            language: Optional language code to filter voices (e.g., 'en', 'ro')

        Returns:
            List of VoiceInfo objects containing voice metadata
        """
        voices = []

        for voice_id, (name, lang, gender, quality, _) in PIPER_VOICES_CATALOG.items():
            # Filter by language if specified
            if language and not lang.startswith(language):
                continue

            # Generate sample URL (will be created later)
            sample_url = f"/static/voice_samples/{voice_id}.mp3"

            voices.append(
                VoiceInfo(
                    id=voice_id,
                    name=name,
                    language=lang,
                    gender=gender,
                    quality=quality,
                    sample_url=sample_url,
                )
            )

        return voices

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
        if voice_id not in PIPER_VOICES_CATALOG:
            raise ValueError(
                f"Unknown voice ID: {voice_id}. "
                f"Available voices: {', '.join(PIPER_VOICES_CATALOG.keys())}"
            )

        name, lang, gender, quality, _ = PIPER_VOICES_CATALOG[voice_id]
        sample_url = f"/static/voice_samples/{voice_id}.mp3"

        return VoiceInfo(
            id=voice_id,
            name=name,
            language=lang,
            gender=gender,
            quality=quality,
            sample_url=sample_url,
        )
