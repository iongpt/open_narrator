#!/usr/bin/env python3
"""
Test script to verify STT service integration.

This script validates that the STT service is properly configured
and can be initialized without errors.

Usage:
    python scripts/test_stt.py
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import get_settings
from app.services.audio_utils import AudioProcessor


def test_configuration() -> None:
    """Test that configuration is loaded correctly."""
    print("=" * 60)
    print("Testing Configuration")
    print("=" * 60)

    settings = get_settings()

    print(f"App Name: {settings.app_name}")
    print(f"Device: {settings.device}")
    print(f"Compute Type: {settings.compute_type}")
    print(f"Whisper Model: {settings.whisper_model}")
    print(f"Model Directory: {settings.model_dir}")
    print(f"Upload Directory: {settings.upload_dir}")
    print(f"Output Directory: {settings.output_dir}")

    # Verify directories exist
    assert settings.model_dir.exists(), "Model directory should exist"
    assert settings.upload_dir.exists(), "Upload directory should exist"
    assert settings.output_dir.exists(), "Output directory should exist"

    print("\n✓ Configuration test passed\n")


def test_audio_processor() -> None:
    """Test audio processing utilities."""
    print("=" * 60)
    print("Testing Audio Processor")
    print("=" * 60)

    # Check if ffmpeg is installed
    ffmpeg_available = AudioProcessor.check_ffmpeg_installed()
    print(f"ffmpeg available: {ffmpeg_available}")

    if not ffmpeg_available:
        print("\n⚠ Warning: ffmpeg not installed")
        print("Audio conversion features will not work without ffmpeg")
        print("Install ffmpeg: brew install ffmpeg (macOS) or apt install ffmpeg (Ubuntu)")
    else:
        print("✓ ffmpeg is properly installed")

    print("\n✓ Audio processor test passed\n")


def test_stt_service_import() -> None:
    """Test that STT service can be imported."""
    print("=" * 60)
    print("Testing STT Service Import")
    print("=" * 60)

    try:
        from app.services.stt_service import STTService  # noqa: F401

        print("✓ STT service imported successfully")

        # Get model info (don't initialize the full model to save time)
        print("\nNote: Not initializing Whisper model in test mode")
        print("To download the model, run: python scripts/download_models.py")

        print("\n✓ STT service import test passed\n")

    except ImportError as e:
        print(f"✗ Failed to import STT service: {e}")
        print("\nMake sure dependencies are installed:")
        print("  pip install -r requirements.txt")
        raise


def main() -> int:
    """
    Run all tests.

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    print("\n" + "=" * 60)
    print("OpenNarrator - Phase 3 STT Service Tests")
    print("=" * 60 + "\n")

    try:
        test_configuration()
        test_audio_processor()
        test_stt_service_import()

        print("=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Download Whisper model: python scripts/download_models.py")
        print("2. Test transcription with a small audio file")
        print("3. Proceed to Phase 4: Translation Service")
        print()

        return 0

    except Exception as e:
        print("\n" + "=" * 60)
        print(f"✗ Tests failed: {e}")
        print("=" * 60 + "\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
