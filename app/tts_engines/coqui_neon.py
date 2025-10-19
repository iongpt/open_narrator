"""Coqui TTS engine for Neon Romanian VITS voice."""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from threading import Lock
from uuid import uuid4

import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import LocalEntryNotFoundError
from pydub import AudioSegment
from TTS.api import TTS

from app.config import get_settings
from app.schemas import VoiceInfo
from app.tts_engines.base import BaseTTSEngine

logger = logging.getLogger(__name__)
settings = get_settings()


class CoquiNeonEngine(BaseTTSEngine):
    """Text-to-speech engine powered by 🐸Coqui TTS Neon Romanian model."""

    ENGINE_NAME = "coqui-neon"
    VOICE_ID = "tts-vits-cv-ro"
    MODEL_NAME = "tts_models/ro/cv/vits"
    HF_REPO_ID = "neongeckocom/tts-vits-cv-ro"
    REQUIRED_FILES = ("config.json", "model_file.pth.tar", "language_ids.json", "speaker_ids.json")

    def __init__(self) -> None:
        self._tts: TTS | None = None
        self._lock = Lock()
        self._cache_dir = settings.model_dir / "coqui_neon"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        # Ensure Coqui downloads are stored inside the project models directory
        os.environ.setdefault("TTS_HOME", str(self._cache_dir))

    # ------------------------------------------------------------------
    # Engine lifecycle helpers
    # ------------------------------------------------------------------
    def _load_tts(self) -> TTS:
        """Load the Coqui TTS model lazily and cache the instance."""
        with self._lock:
            if self._tts is None:
                logger.info("Loading Coqui Neon Romanian model (%s)", self.MODEL_NAME)
                gpu_enabled = torch.cuda.is_available()
                self._tts = TTS(
                    model_name=self.MODEL_NAME,
                    progress_bar=False,
                    gpu=gpu_enabled,
                )
            return self._tts

    # ------------------------------------------------------------------
    # BaseTTSEngine implementation
    # ------------------------------------------------------------------
    def generate_audio(
        self,
        text: str,
        voice_id: str,
        language: str,
        *,
        length_scale: float | None = None,
        noise_scale: float | None = None,
        noise_w_scale: float | None = None,
    ) -> str:
        if not text:
            raise ValueError("Text cannot be empty")
        if voice_id != self.VOICE_ID:
            raise ValueError(f"Unknown Coqui Neon voice: {voice_id}")
        if not language.startswith("ro"):
            logger.warning(
                "Coqui Neon voice is tuned for Romanian; requested language=%s", language
            )

        logger.info("Generating Romanian audio with Coqui Neon model (%s)", voice_id)

        tts = self._load_tts()

        # Length scale can be mapped to speech speed (inverse relation)
        speed = None
        if length_scale is not None:
            try:
                speed = max(0.2, min(3.0, 1.0 / length_scale))
            except ZeroDivisionError:  # Defensive guard
                speed = None

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_file:
            wav_path = Path(wav_file.name)

        tts_kwargs = {
            "text": text,
            "file_path": str(wav_path),
            "split_sentences": True,
        }

        if speed is not None:
            tts_kwargs["speed"] = speed

        if getattr(tts, "is_multi_lingual", False):  # Some releases expose languages
            tts_kwargs["language"] = "ro"

        tts.tts_to_file(**tts_kwargs)

        mp3_filename = f"coqui_{uuid4().hex}.mp3"
        mp3_path = settings.output_dir / mp3_filename

        audio = AudioSegment.from_file(wav_path, format="wav")
        audio.export(mp3_path, format="mp3")
        wav_path.unlink(missing_ok=True)

        logger.info("Generated Coqui Neon audio: %s", mp3_path)
        return str(mp3_path)

    def list_voices(self, language: str | None = None) -> list[VoiceInfo]:
        if language and not language.startswith("ro"):
            return []

        return [
            VoiceInfo(
                id=self.VOICE_ID,
                name="Neon VITS Romanian",
                language="ro",
                gender="female",
                quality="studio",
                sample_url=None,
            )
        ]

    def get_voice_info(self, voice_id: str) -> VoiceInfo:
        if voice_id != self.VOICE_ID:
            raise ValueError(f"Unknown Coqui Neon voice: {voice_id}")
        return VoiceInfo(
            id=self.VOICE_ID,
            name="Neon VITS Romanian",
            language="ro",
            gender="female",
            quality="studio",
            sample_url=None,
        )

    def is_voice_available(self, voice_id: str) -> bool:
        if voice_id != self.VOICE_ID:
            return False
        try:
            hf_hub_download(
                repo_id=self.HF_REPO_ID,
                filename="model_file.pth.tar",
                cache_dir=str(self._cache_dir),
                local_files_only=True,
            )
            return True
        except LocalEntryNotFoundError:
            return False

    def download_voice(self, voice_id: str) -> None:
        if voice_id != self.VOICE_ID:
            raise ValueError(f"Unknown Coqui Neon voice: {voice_id}")

        for filename in self.REQUIRED_FILES:
            logger.info("Ensuring Coqui Neon model asset %s", filename)
            hf_hub_download(
                repo_id=self.HF_REPO_ID,
                filename=filename,
                cache_dir=str(self._cache_dir),
            )

        # Clear cached TTS instance so that it picks up the freshly downloaded files
        with self._lock:
            self._tts = None
