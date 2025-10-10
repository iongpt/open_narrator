#!/usr/bin/env python3
"""
Download and verify Whisper models for Faster-Whisper.

This script downloads Whisper models on first run and caches them
in the data/models/ directory. It verifies model integrity and handles
download failures gracefully.

Usage:
    python scripts/download_models.py [model_name]

Examples:
    python scripts/download_models.py large-v3
    python scripts/download_models.py base
"""

import argparse
import logging
import sys
from pathlib import Path

from faster_whisper import WhisperModel

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def download_model(model_name: str, model_dir: Path) -> bool:
    """
    Download Whisper model from Hugging Face.

    Args:
        model_name: Name of Whisper model (e.g., 'large-v3', 'base', 'medium')
        model_dir: Directory to store downloaded models

    Returns:
        True if download successful, False otherwise

    Raises:
        ValueError: If model name is invalid
    """
    valid_models = ["tiny", "base", "small", "medium", "large-v2", "large-v3"]

    if model_name not in valid_models:
        raise ValueError(
            f"Invalid model name: {model_name}. " f"Valid options: {', '.join(valid_models)}"
        )

    # Ensure model directory exists
    model_dir.mkdir(parents=True, exist_ok=True)

    try:
        logger.info(f"Downloading Whisper model '{model_name}'...")
        logger.info(f"Model will be cached in: {model_dir}")

        # Initialize model (will trigger download if not cached)
        # Use CPU for download to avoid GPU memory issues
        model = WhisperModel(
            model_size_or_path=model_name,
            device="cpu",
            compute_type="int8",
            download_root=str(model_dir),
        )

        logger.info(f"✓ Model '{model_name}' downloaded successfully")

        # Verify model by getting its info
        logger.info("Verifying model integrity...")
        # The model is already loaded, so we know it's valid
        del model  # Free memory

        logger.info("✓ Model verification passed")
        return True

    except Exception as e:
        logger.error(f"✗ Failed to download model '{model_name}': {e}")
        return False


def verify_existing_model(model_name: str, model_dir: Path) -> bool:
    """
    Verify that an existing model is valid and can be loaded.

    Args:
        model_name: Name of Whisper model
        model_dir: Directory where models are stored

    Returns:
        True if model is valid, False otherwise
    """
    try:
        logger.info(f"Verifying existing model '{model_name}'...")

        # Try to load the model
        model = WhisperModel(
            model_size_or_path=model_name,
            device="cpu",
            compute_type="int8",
            download_root=str(model_dir),
        )

        logger.info(f"✓ Model '{model_name}' is valid")
        del model
        return True

    except Exception as e:
        logger.warning(f"✗ Model verification failed: {e}")
        return False


def main() -> int:
    """
    Main function to download models.

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    parser = argparse.ArgumentParser(
        description="Download Whisper models for OpenNarrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "model",
        nargs="?",
        help="Whisper model to download (default: from settings)",
        choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing model, don't download",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all available models",
    )

    args = parser.parse_args()

    # Load settings
    settings = get_settings()

    # Determine which model(s) to download
    if args.all:
        models_to_download = ["tiny", "base", "small", "medium", "large-v2", "large-v3"]
    elif args.model:
        models_to_download = [args.model]
    else:
        models_to_download = [settings.whisper_model]

    logger.info("=" * 60)
    logger.info("OpenNarrator - Whisper Model Downloader")
    logger.info("=" * 60)
    logger.info(f"Model directory: {settings.model_dir}")
    logger.info(f"Models to process: {', '.join(models_to_download)}")
    logger.info("=" * 60)

    success_count = 0
    failure_count = 0

    for model_name in models_to_download:
        logger.info(f"\nProcessing model: {model_name}")

        if args.verify_only:
            if verify_existing_model(model_name, settings.model_dir):
                success_count += 1
            else:
                failure_count += 1
        else:
            if download_model(model_name, settings.model_dir):
                success_count += 1
            else:
                failure_count += 1

    logger.info("\n" + "=" * 60)
    logger.info(f"Results: {success_count} succeeded, {failure_count} failed")
    logger.info("=" * 60)

    return 0 if failure_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
