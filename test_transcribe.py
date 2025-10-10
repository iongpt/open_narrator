#!/usr/bin/env python3
"""
Simple script to test transcription inline without background jobs.

Usage:
    python test_transcribe.py [path/to/audio.mp3]

If no path is provided, uses tests/sample/sample.mp3
"""

import asyncio
import sys
from pathlib import Path

# Add app directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from app.services.stt_service import STTService


async def test_transcribe(audio_path: str) -> None:
    """
    Test transcription of an audio file.

    Args:
        audio_path: Path to the audio file to transcribe
    """
    print(f"🎵 Testing transcription for: {audio_path}\n")

    # Check if file exists
    file_path = Path(audio_path)
    if not file_path.exists():
        print(f"❌ Error: File not found: {audio_path}")
        sys.exit(1)

    # Get file size
    file_size_mb = file_path.stat().st_size / (1024 * 1024)
    print(f"📁 File size: {file_size_mb:.2f} MB")

    try:
        # Initialize STT service
        print("\n🔧 Initializing Whisper model...")
        stt_service = STTService()

        # Get model info
        model_info = stt_service.get_model_info()
        print(f"   Model: {model_info['model_name']}")
        print(f"   Device: {model_info['device']}")
        print(f"   Compute type: {model_info['compute_type']}")

        # Validate audio file
        print("\n✅ Validating audio file...")
        audio_info = stt_service.validate_audio(audio_path)
        print(f"   Format: {audio_info['format']}")
        print(f"   Size: {audio_info['size_mb']} MB")

        # Transcribe
        print("\n🎙️  Starting transcription...")
        print("   (This may take a few minutes depending on file size)\n")

        transcript = await stt_service.transcribe(
            file_path=audio_path,
            language="en",  # Change this if your audio is in a different language
        )

        # Display results
        print("\n" + "=" * 80)
        print("✨ TRANSCRIPTION COMPLETE")
        print("=" * 80)
        print(f"\n📝 Transcript ({len(transcript)} characters):\n")
        print(transcript)
        print("\n" + "=" * 80)

        # Save to file
        output_file = file_path.with_suffix(".txt")
        output_file.write_text(transcript)
        print(f"\n💾 Transcript saved to: {output_file}")

    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"\n❌ Validation Error: {e}")
        sys.exit(1)
    except RuntimeError as e:
        print(f"\n❌ Runtime Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


def main() -> None:
    """Main entry point."""
    # Get audio file path from command line or use default
    audio_path = sys.argv[1] if len(sys.argv) > 1 else "tests/sample/sample.mp3"

    # Run async transcription
    asyncio.run(test_transcribe(audio_path))


if __name__ == "__main__":
    main()
