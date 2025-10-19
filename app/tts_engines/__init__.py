"""TTS engine registry for OpenNarrator."""

from __future__ import annotations

from collections.abc import Mapping

from app.tts_engines.base import BaseTTSEngine
from app.tts_engines.coqui_neon import CoquiNeonEngine
from app.tts_engines.mms import MMSTTSEngine
from app.tts_engines.piper import PiperEngine

ENGINE_REGISTRY: Mapping[str, type[BaseTTSEngine]] = {
    "piper": PiperEngine,
    "coqui-neon": CoquiNeonEngine,
    "mms": MMSTTSEngine,
}

ENGINE_LABELS: Mapping[str, str] = {
    "piper": "Piper",
    "coqui-neon": "Coqui VITS",
    "mms": "Meta MMS",
}
