"""Meta MMS TTS engine for Romanian (mms-tts-ron)."""

from __future__ import annotations

import logging
import tempfile
import wave
from pathlib import Path
from threading import Lock
from uuid import uuid4

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import LocalEntryNotFoundError
from pydub import AudioSegment
from transformers import AutoTokenizer, VitsModel

from app.config import get_settings
from app.schemas import VoiceInfo
from app.tts_engines.base import BaseTTSEngine

logger = logging.getLogger(__name__)
settings = get_settings()


class MMSTTSEngine(BaseTTSEngine):
    """Meta MMS text-to-speech engine using the Romanian VITS model."""

    ENGINE_NAME = "mms"
    VOICE_ID = "mms-tts-ron"
    MODEL_ID = "facebook/mms-tts-ron"
    REQUIRED_FILES = (
        "config.json",
        "model.safetensors",
        "tokenizer_config.json",
        "vocab.json",
        "special_tokens_map.json",
    )

    def __init__(self) -> None:
        self._model: VitsModel | None = None
        self._tokenizer: AutoTokenizer | None = None
        self._lock = Lock()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._cache_dir = settings.model_dir / "mms_tts"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _ensure_model(self, *, preload_only: bool = False) -> None:
        """Load MMS model/tokenizer with caching and optional preload mode."""
        with self._lock:
            if preload_only:
                logger.info("Prefetching Meta MMS assets to %s", self._cache_dir)
            if self._model is None or preload_only:
                model = VitsModel.from_pretrained(
                    self.MODEL_ID,
                    cache_dir=str(self._cache_dir),
                )
                tokenizer = AutoTokenizer.from_pretrained(
                    self.MODEL_ID,
                    cache_dir=str(self._cache_dir),
                )
                if preload_only:
                    return
                self._model = model.to(self._device)
                self._tokenizer = tokenizer

    def _get_model(self) -> tuple[VitsModel, AutoTokenizer]:
        if self._model is None or self._tokenizer is None:
            self._ensure_model()
        assert self._model is not None and self._tokenizer is not None
        return self._model, self._tokenizer

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
            raise ValueError(f"Unknown MMS voice: {voice_id}")
        if not language.startswith("ro"):
            logger.warning("MMS Romanian voice received language=%s", language)

        model, tokenizer = self._get_model()

        # Map generic slider controls to MMS-specific synthesis knobs
        original_speaking_rate = getattr(model, "speaking_rate", None)
        original_noise_scale = getattr(model, "noise_scale", None)
        original_noise_scale_duration = getattr(model, "noise_scale_duration", None)

        try:
            if length_scale is not None:
                # MMS expects speaking_rate > 0.0 where 1.0 is default. Invert length_scale to
                # match Piper-style semantics (smaller length_scale => faster speech).
                speaking_rate = 1.0 / max(length_scale, 1e-3)
                model.speaking_rate = float(min(max(speaking_rate, 0.25), 4.0))
            elif original_speaking_rate is None and hasattr(model.config, "speaking_rate"):
                model.speaking_rate = float(model.config.speaking_rate)

            if noise_scale is not None and hasattr(model, "noise_scale"):
                model.noise_scale = float(min(max(noise_scale, 0.0), 2.0))
            elif original_noise_scale is None and hasattr(model.config, "noise_scale"):
                model.noise_scale = float(model.config.noise_scale)

            if noise_w_scale is not None and hasattr(model, "noise_scale_duration"):
                model.noise_scale_duration = float(min(max(noise_w_scale, 0.0), 2.0))
            elif original_noise_scale_duration is None and hasattr(
                model.config, "noise_scale_duration"
            ):
                model.noise_scale_duration = float(model.config.noise_scale_duration)

        except Exception as exc:  # pragma: no cover - protective logging
            logger.warning("Failed to apply MMS prosody controls: %s", exc)

        inputs = tokenizer(text, return_tensors="pt")
        inputs = inputs.to(self._device)

        logger.info("Generating Romanian audio with Meta MMS model (%s)", voice_id)
        try:
            with torch.no_grad():
                outputs = model(**inputs)
        finally:
            if original_speaking_rate is not None and hasattr(model, "speaking_rate"):
                model.speaking_rate = original_speaking_rate
            if original_noise_scale is not None and hasattr(model, "noise_scale"):
                model.noise_scale = original_noise_scale
            if original_noise_scale_duration is not None and hasattr(model, "noise_scale_duration"):
                model.noise_scale_duration = original_noise_scale_duration

        waveform = outputs.waveform.squeeze(0).cpu().numpy()
        sampling_rate = model.config.sampling_rate

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_file:
            wav_path = Path(wav_file.name)

        self._write_pcm_to_wav(waveform, sampling_rate, wav_path)

        mp3_filename = f"mms_{uuid4().hex}.mp3"
        mp3_path = settings.output_dir / mp3_filename

        audio = AudioSegment.from_file(wav_path, format="wav")
        audio.export(mp3_path, format="mp3")
        wav_path.unlink(missing_ok=True)

        logger.info("Generated MMS Romanian audio: %s", mp3_path)

        return str(mp3_path)

    def list_voices(self, language: str | None = None) -> list[VoiceInfo]:
        if language and not language.startswith("ro"):
            return []
        return [
            VoiceInfo(
                id=self.VOICE_ID,
                name="MMS Romanian",
                language="ro",
                gender="female",
                quality="research",
                sample_url=None,
            )
        ]

    def get_voice_info(self, voice_id: str) -> VoiceInfo:
        if voice_id != self.VOICE_ID:
            raise ValueError(f"Unknown MMS voice: {voice_id}")
        return VoiceInfo(
            id=self.VOICE_ID,
            name="MMS Romanian",
            language="ro",
            gender="female",
            quality="research",
            sample_url=None,
        )

    def is_voice_available(self, voice_id: str) -> bool:
        if voice_id != self.VOICE_ID:
            return False
        try:
            hf_hub_download(
                repo_id=self.MODEL_ID,
                filename="config.json",
                cache_dir=str(self._cache_dir),
                local_files_only=True,
            )
            return True
        except LocalEntryNotFoundError:
            return False

    def download_voice(self, voice_id: str) -> None:
        if voice_id != self.VOICE_ID:
            raise ValueError(f"Unknown MMS voice: {voice_id}")

        for filename in self.REQUIRED_FILES:
            logger.info("Ensuring MMS model asset %s", filename)
            hf_hub_download(
                repo_id=self.MODEL_ID,
                filename=filename,
                cache_dir=str(self._cache_dir),
            )

        self._ensure_model(preload_only=True)
        with self._lock:
            self._model = None
            self._tokenizer = None

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _write_pcm_to_wav(waveform: np.ndarray, sampling_rate: int, wav_path: Path) -> None:
        clipped = np.clip(waveform, -1.0, 1.0)
        pcm_data = (clipped * np.iinfo(np.int16).max).astype(np.int16)

        with wave.open(str(wav_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)  # 16-bit audio
            wav_file.setframerate(sampling_rate)
            wav_file.writeframes(pcm_data.tobytes())
