"""Utility service for preparing long-form text for TTS narration."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(slots=True)
class TextPreprocessor:
    """Heuristically prepare book-length text for more natural narration."""

    min_sentence_length: int = 40

    def prepare_for_tts(self, text: str) -> str:
        """Normalize whitespace and ensure paragraphs end with punctuation."""

        if not text or not text.strip():
            return text

        paragraphs = self._split_paragraphs(text)
        cleaned_paragraphs: list[str] = []

        for paragraph in paragraphs:
            normalized = self._normalize_paragraph(paragraph)
            if normalized:
                cleaned_paragraphs.append(normalized)

        return "\n\n".join(cleaned_paragraphs)

    def _split_paragraphs(self, text: str) -> list[str]:
        parts = re.split(r"\n{2,}", text)
        return [p for p in (part.strip() for part in parts) if p]

    def _normalize_paragraph(self, paragraph: str) -> str:
        lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
        if not lines:
            return ""

        paragraph_text = " ".join(lines)
        paragraph_text = re.sub(r"\s+", " ", paragraph_text)

        sentences = re.split(r"(?<=[.!?…])\s+", paragraph_text)
        processed: list[str] = []

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            if sentence[-1] not in ".!?…" and len(sentence) >= self.min_sentence_length:
                sentence = f"{sentence}."

            processed.append(sentence)

        return " ".join(processed)
