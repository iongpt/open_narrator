#!/usr/bin/env python3
"""Voice management script for Piper TTS.

This script helps download and manage Piper voice models from Hugging Face.
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.tts_engines.piper import PIPER_VOICES_CATALOG, PiperEngine


def list_voices(language: str | None = None) -> None:
    """
    List all available Piper voices.

    Args:
        language: Optional language filter
    """
    print("\n=== Available Piper Voices ===\n")

    # Group by language
    voices_by_lang: dict[str, list[tuple[str, tuple[str, str, str, str, str]]]] = {}

    for voice_id, voice_info in PIPER_VOICES_CATALOG.items():
        lang = voice_info[1]

        # Filter by language if specified
        if language and not lang.startswith(language):
            continue

        if lang not in voices_by_lang:
            voices_by_lang[lang] = []

        voices_by_lang[lang].append((voice_id, voice_info))

    # Print grouped by language
    for lang in sorted(voices_by_lang.keys()):
        lang_name = {
            "en": "English",
            "ro": "Romanian",
            "es": "Spanish",
            "fr": "French",
            "de": "German",
            "it": "Italian",
            "pt": "Portuguese",
            "nl": "Dutch",
            "pl": "Polish",
            "ru": "Russian",
            "uk": "Ukrainian",
            "ja": "Japanese",
            "zh": "Chinese",
            "ar": "Arabic",
            "tr": "Turkish",
            "hi": "Hindi",
            "ko": "Korean",
        }.get(lang, lang.upper())

        print(f"\n{lang_name} ({lang}):")
        print("-" * 60)

        for voice_id, (name, _, gender, quality, _) in voices_by_lang[lang]:
            print(f"  {voice_id}")
            print(f"    Name:    {name}")
            print(f"    Gender:  {gender}")
            print(f"    Quality: {quality}")
            print()

    total_voices = sum(len(voices) for voices in voices_by_lang.values())
    print(f"\nTotal voices: {total_voices}")

    if language:
        print(f"(filtered by language: {language})")


def download_voice(voice_id: str) -> None:
    """
    Download a specific voice model.

    Args:
        voice_id: Voice identifier to download
    """
    engine = PiperEngine()

    if engine.is_voice_available(voice_id):
        print(f"✓ Voice '{voice_id}' is already downloaded")
        return

    print(f"Downloading voice: {voice_id}")

    try:
        engine.download_voice(voice_id)
        print(f"✓ Successfully downloaded: {voice_id}")

    except ValueError as e:
        print(f"✗ Error: {e}", file=sys.stderr)
        sys.exit(1)

    except RuntimeError as e:
        print(f"✗ Download failed: {e}", file=sys.stderr)
        sys.exit(1)


def download_language(language: str) -> None:
    """
    Download all voices for a specific language.

    Args:
        language: Language code (e.g., 'en', 'ro')
    """
    engine = PiperEngine()

    # Find all voices for this language
    voices_to_download = [
        voice_id
        for voice_id, voice_info in PIPER_VOICES_CATALOG.items()
        if voice_info[1].startswith(language)
    ]

    if not voices_to_download:
        print(f"✗ No voices found for language: {language}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(voices_to_download)} voice(s) for language '{language}'")

    for voice_id in voices_to_download:
        if engine.is_voice_available(voice_id):
            print(f"  ✓ {voice_id} (already downloaded)")
        else:
            print(f"  Downloading {voice_id}...", end=" ")
            try:
                engine.download_voice(voice_id)
                print("✓")
            except Exception as e:
                print(f"✗ Failed: {e}", file=sys.stderr)

    print(f"\n✓ Completed download for language: {language}")


def check_voice(voice_id: str) -> None:
    """
    Check if a voice is available locally.

    Args:
        voice_id: Voice identifier to check
    """
    engine = PiperEngine()

    if voice_id not in PIPER_VOICES_CATALOG:
        print(f"✗ Unknown voice ID: {voice_id}", file=sys.stderr)
        print(f"\nRun 'python {sys.argv[0]} list' to see available voices")
        sys.exit(1)

    if engine.is_voice_available(voice_id):
        print(f"✓ Voice '{voice_id}' is available")
        voice_info = engine.get_voice_info(voice_id)
        print(f"  Name:     {voice_info.name}")
        print(f"  Language: {voice_info.language}")
        print(f"  Gender:   {voice_info.gender}")
        print(f"  Quality:  {voice_info.quality}")
    else:
        print(f"✗ Voice '{voice_id}' is not downloaded")
        print(f"\nTo download, run: python {sys.argv[0]} download {voice_id}")


def generate_sample(voice_id: str, text: str | None = None) -> None:
    """
    Generate a sample audio file for a voice.

    Args:
        voice_id: Voice identifier
        text: Optional custom text (default: language-specific sample)
    """
    engine = PiperEngine()

    if voice_id not in PIPER_VOICES_CATALOG:
        print(f"✗ Unknown voice ID: {voice_id}", file=sys.stderr)
        sys.exit(1)

    # Default sample texts by language
    default_samples = {
        "en": "Welcome to OpenNarrator. This is a sample of the voice.",
        "ro": "Bun venit la OpenNarrator. Acesta este un exemplu de voce.",
        "es": "Bienvenido a OpenNarrator. Esta es una muestra de la voz.",
        "fr": "Bienvenue sur OpenNarrator. Ceci est un échantillon de la voix.",
        "de": "Willkommen bei OpenNarrator. Dies ist eine Stimmprobe.",
        "it": "Benvenuto su OpenNarrator. Questo è un campione della voce.",
        "pt": "Bem-vindo ao OpenNarrator. Este é um exemplo da voz.",
        "nl": "Welkom bij OpenNarrator. Dit is een voorbeeld van de stem.",
        "pl": "Witamy w OpenNarrator. To jest próbka głosu.",
        "ru": "Добро пожаловать в OpenNarrator. Это образец голоса.",
        "uk": "Ласкаво просимо до OpenNarrator. Це зразок голосу.",
        "ja": "OpenNarratorへようこそ。これは音声のサンプルです。",
        "zh": "欢迎来到OpenNarrator。这是语音样本。",
        "ar": "مرحبا بكم في OpenNarrator. هذا نموذج من الصوت.",
        "tr": "OpenNarrator'a hoş geldiniz. Bu, sesin bir örneğidir.",
        "hi": "OpenNarrator में आपका स्वागत है। यह आवाज़ का एक नमूना है।",
        "ko": "OpenNarrator에 오신 것을 환영합니다. 이것은 음성 샘플입니다.",
    }

    # Get voice language
    voice_lang = PIPER_VOICES_CATALOG[voice_id][1]

    # Use custom text or default for language
    sample_text = text or default_samples.get(voice_lang, default_samples["en"])

    print(f"Generating sample for voice: {voice_id}")
    print(f"Text: {sample_text}")

    try:
        # Ensure voice is downloaded
        if not engine.is_voice_available(voice_id):
            print("Voice not downloaded. Downloading now...")
            engine.download_voice(voice_id)

        # Generate audio
        output_path = engine.generate_audio(sample_text, voice_id, voice_lang)
        print(f"✓ Sample generated: {output_path}")

    except Exception as e:
        print(f"✗ Failed to generate sample: {e}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    """Main entry point for voice management CLI."""
    parser = argparse.ArgumentParser(
        description="Manage Piper TTS voice models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all available voices
  python setup_voices.py list

  # List voices for a specific language
  python setup_voices.py list --language ro

  # Download a specific voice
  python setup_voices.py download en_US-lessac-medium

  # Download all voices for a language
  python setup_voices.py download-lang ro

  # Check if a voice is available
  python setup_voices.py check ro_RO-mihai-medium

  # Generate a sample audio
  python setup_voices.py sample en_US-lessac-medium
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # List command
    list_parser = subparsers.add_parser("list", help="List available voices")
    list_parser.add_argument(
        "--language",
        "-l",
        help="Filter by language code (e.g., en, ro, es)",
    )

    # Download command
    download_parser = subparsers.add_parser("download", help="Download a voice model")
    download_parser.add_argument("voice_id", help="Voice ID to download")

    # Download language command
    download_lang_parser = subparsers.add_parser(
        "download-lang", help="Download all voices for a language"
    )
    download_lang_parser.add_argument("language", help="Language code (e.g., en, ro, es)")

    # Check command
    check_parser = subparsers.add_parser("check", help="Check if a voice is available")
    check_parser.add_argument("voice_id", help="Voice ID to check")

    # Sample command
    sample_parser = subparsers.add_parser("sample", help="Generate a sample audio file")
    sample_parser.add_argument("voice_id", help="Voice ID to use")
    sample_parser.add_argument("--text", help="Custom text to synthesize")

    args = parser.parse_args()

    # Execute command
    if args.command == "list":
        list_voices(args.language)
    elif args.command == "download":
        download_voice(args.voice_id)
    elif args.command == "download-lang":
        download_language(args.language)
    elif args.command == "check":
        check_voice(args.voice_id)
    elif args.command == "sample":
        generate_sample(args.voice_id, args.text)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
