"""Utility helpers for loading and saving bulk processing presets."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from app.config import get_settings
from app.schemas import BulkPreset

logger = logging.getLogger(__name__)


def load_bulk_preset() -> BulkPreset | None:
    """Load the bulk processing preset from disk."""

    settings = get_settings()
    preset_path = settings.bulk_preset_path

    if not preset_path.exists():
        logger.debug("Bulk preset file does not exist: %s", preset_path)
        return None

    try:
        data = json.loads(preset_path.read_text(encoding="utf-8"))
        preset = BulkPreset.model_validate(data)
        logger.debug("Loaded bulk preset from %s", preset_path)
        return preset
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error("Failed to load bulk preset: %s", exc)
        return None


def save_bulk_preset(preset: BulkPreset) -> None:
    """Persist the bulk processing preset to disk atomically."""

    settings = get_settings()
    preset_path = settings.bulk_preset_path
    preset_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = preset_path.with_suffix(".tmp")
    tmp_path.write_text(preset.model_dump_json(indent=2), encoding="utf-8")
    tmp_path.replace(preset_path)
    logger.info("Saved bulk preset to %s", preset_path)


def ensure_output_directory(path: str) -> Path:
    """Ensure the output directory exists and return its Path object."""

    output_path = Path(path)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path
