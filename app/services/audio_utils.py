"""Audio processing utilities for OpenNarrator."""

import logging
import subprocess
from pathlib import Path

from pydub import AudioSegment

logger = logging.getLogger(__name__)


class AudioProcessor:
    """
    Utility class for audio file processing and conversion.

    Handles audio format conversion, validation, and metadata extraction
    using pydub and ffmpeg.
    """

    @staticmethod
    def convert_to_wav(
        input_path: str | Path,
        output_path: str | Path | None = None,
        sample_rate: int = 16000,
    ) -> Path:
        """
        Convert audio file to WAV format.

        Whisper works best with 16kHz mono WAV files. This function converts
        any audio format supported by ffmpeg to WAV with optimal settings.

        Args:
            input_path: Path to input audio file (MP3, M4A, etc.)
            output_path: Optional path for output WAV file (defaults to input_path.wav)
            sample_rate: Target sample rate in Hz (default: 16000 for Whisper)

        Returns:
            Path to the converted WAV file

        Raises:
            FileNotFoundError: If input file doesn't exist
            RuntimeError: If conversion fails
        """
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Audio file not found: {input_path}")

        output_path = input_path.with_suffix(".wav") if output_path is None else Path(output_path)

        try:
            logger.info(f"Converting {input_path} to WAV format...")

            # Load audio file with pydub (supports many formats via ffmpeg)
            audio = AudioSegment.from_file(str(input_path))

            # Convert to mono
            audio = audio.set_channels(1)

            # Set sample rate
            audio = audio.set_frame_rate(sample_rate)

            # Export as WAV
            audio.export(str(output_path), format="wav")

            logger.info(f"✓ Converted to WAV: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Failed to convert audio to WAV: {e}")
            raise RuntimeError(f"Audio conversion failed: {e}") from e

    @staticmethod
    def get_audio_info(file_path: str | Path) -> dict[str, str | int | float]:
        """
        Get audio file metadata using ffprobe.

        Args:
            file_path: Path to audio file

        Returns:
            Dictionary with audio metadata (duration, channels, sample_rate, etc.)

        Raises:
            FileNotFoundError: If file doesn't exist
            RuntimeError: If metadata extraction fails
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        try:
            # Use pydub to get basic info (more reliable than ffprobe)
            audio = AudioSegment.from_file(str(file_path))

            duration_seconds = len(audio) / 1000.0  # pydub uses milliseconds

            metadata = {
                "duration_seconds": round(duration_seconds, 2),
                "duration_formatted": AudioProcessor._format_duration(duration_seconds),
                "channels": audio.channels,
                "sample_rate": audio.frame_rate,
                "sample_width": audio.sample_width,
                "frame_count": audio.frame_count(),
                "size_mb": round(file_path.stat().st_size / (1024 * 1024), 2),
            }

            logger.info(f"Audio info: {metadata['duration_formatted']}, {metadata['channels']}ch")
            return metadata

        except Exception as e:
            logger.error(f"Failed to extract audio metadata: {e}")
            raise RuntimeError(f"Metadata extraction failed: {e}") from e

    @staticmethod
    def _format_duration(seconds: float) -> str:
        """
        Format duration in seconds to HH:MM:SS.

        Args:
            seconds: Duration in seconds

        Returns:
            Formatted string (e.g., "01:23:45")
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    @staticmethod
    def validate_audio_file(file_path: str | Path) -> bool:
        """
        Validate that a file is a valid audio file.

        Args:
            file_path: Path to audio file

        Returns:
            True if file is valid audio, False otherwise
        """
        file_path = Path(file_path)

        if not file_path.exists():
            logger.warning(f"File does not exist: {file_path}")
            return False

        if file_path.stat().st_size == 0:
            logger.warning(f"File is empty: {file_path}")
            return False

        try:
            # Try to load the file with pydub
            audio = AudioSegment.from_file(str(file_path))

            # Check if audio has content
            if len(audio) == 0:
                logger.warning(f"Audio file has zero duration: {file_path}")
                return False

            logger.info(f"✓ Audio file is valid: {file_path}")
            return True

        except Exception as e:
            logger.warning(f"Invalid audio file {file_path}: {e}")
            return False

    @staticmethod
    def check_ffmpeg_installed() -> bool:
        """
        Check if ffmpeg is installed and available.

        Returns:
            True if ffmpeg is available, False otherwise
        """
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                logger.info("✓ ffmpeg is installed")
                return True
            else:
                logger.warning("✗ ffmpeg not found")
                return False
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.warning("✗ ffmpeg not found or not responding")
            return False

    @staticmethod
    def normalize_audio(
        input_path: str | Path,
        output_path: str | Path | None = None,
        target_dbfs: float = -20.0,
    ) -> Path:
        """
        Normalize audio volume to a target dBFS level.

        Args:
            input_path: Path to input audio file
            output_path: Optional path for output file (defaults to input_path_normalized.ext)
            target_dbfs: Target dBFS level (default: -20.0, good for speech)

        Returns:
            Path to normalized audio file

        Raises:
            FileNotFoundError: If input file doesn't exist
            RuntimeError: If normalization fails
        """
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Audio file not found: {input_path}")

        output_path = (
            input_path.with_stem(f"{input_path.stem}_normalized")
            if output_path is None
            else Path(output_path)
        )

        try:
            logger.info(f"Normalizing audio to {target_dbfs} dBFS...")

            # Load audio
            audio = AudioSegment.from_file(str(input_path))

            # Calculate gain needed to reach target dBFS
            change_in_dbfs = target_dbfs - audio.dBFS
            normalized_audio = audio.apply_gain(change_in_dbfs)

            # Export normalized audio
            normalized_audio.export(
                str(output_path),
                format=input_path.suffix[1:],  # Remove the dot from extension
            )

            logger.info(f"✓ Normalized audio saved to: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Failed to normalize audio: {e}")
            raise RuntimeError(f"Audio normalization failed: {e}") from e
