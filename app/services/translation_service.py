"""Translation service orchestrating text chunking and LLM translation."""

import logging
from collections.abc import Callable
from typing import Any

from app.config import get_settings
from app.providers.anthropic import AnthropicProvider
from app.providers.base import BaseLLMProvider
from app.services.chunking_service import ChunkingService

logger = logging.getLogger(__name__)


class TranslationService:
    """
    Service for translating long texts with intelligent chunking.

    This service orchestrates:
    1. Text chunking with paragraph preservation
    2. Translation of each chunk via LLM provider
    3. Progress tracking during translation
    4. Reassembly of translated chunks

    Features:
    - Supports multiple LLM providers via factory pattern
    - Smart chunking to avoid splitting paragraphs
    - Context-aware translation prompts
    - Progress callbacks for UI updates
    - Error handling with partial progress recovery
    """

    def __init__(
        self,
        provider: BaseLLMProvider | None = None,
        chunking_service: ChunkingService | None = None,
    ):
        """
        Initialize the translation service.

        Args:
            provider: LLM provider to use (defaults to Anthropic)
            chunking_service: Chunking service instance (defaults to new instance)
        """
        self.settings = get_settings()
        self.provider = provider or self._get_default_provider()
        self.chunking_service = chunking_service or ChunkingService()
        logger.info(f"Initialized translation service with provider: {self.provider}")

    def _get_default_provider(self) -> BaseLLMProvider:
        """
        Get the default LLM provider based on settings.

        Returns:
            Default LLM provider instance
        """
        # For now, only Anthropic is implemented
        # Future: Add factory logic to support multiple providers
        return AnthropicProvider()

    async def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        context: str = "",
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> str:
        """
        Translate long text with automatic chunking and progress tracking.

        This method:
        1. Chunks the text if it exceeds max_tokens
        2. Translates each chunk sequentially
        3. Calls progress_callback after each chunk (if provided)
        4. Reassembles translated chunks preserving structure

        Args:
            text: The text to translate
            source_lang: Source language code (e.g., "en")
            target_lang: Target language code (e.g., "ro")
            context: Optional context for translation (e.g., "mystery audiobook")
            progress_callback: Optional callback function(current_chunk, total_chunks, status_message)

        Returns:
            The complete translated text

        Raises:
            ValueError: If inputs are invalid
            RuntimeError: If translation fails
        """
        if not text or not text.strip():
            raise ValueError("Text to translate cannot be empty")

        if not source_lang or not target_lang:
            raise ValueError("Both source_lang and target_lang are required")

        logger.info(
            f"Starting translation: {source_lang} -> {target_lang}, "
            f"{len(text)} chars, context: {context or 'none'}"
        )

        # Chunk the text
        chunks = self.chunking_service.chunk_text(
            text,
            max_tokens=self.settings.translation_max_tokens,
            preserve_paragraphs=True,
            overlap_tokens=self.settings.translation_chunk_overlap,
        )

        total_chunks = len(chunks)
        logger.info(f"Text split into {total_chunks} chunk(s)")

        if progress_callback:
            progress_callback(0, total_chunks, "Starting translation...")

        # Translate each chunk
        translated_chunks: list[str] = []
        for i, chunk in enumerate(chunks):
            chunk_num = i + 1
            logger.info(f"Translating chunk {chunk_num}/{total_chunks}")

            if progress_callback:
                progress_callback(
                    chunk_num - 1,
                    total_chunks,
                    f"Translating chunk {chunk_num}/{total_chunks}...",
                )

            try:
                # Translate the chunk
                translated_chunk = await self.provider.translate(
                    text=chunk,
                    source_lang=source_lang,
                    target_lang=target_lang,
                    context=context,
                )

                # Remove overlap markers from translation if present
                translated_chunk = self._clean_translation(translated_chunk)

                translated_chunks.append(translated_chunk)

                logger.info(f"Chunk {chunk_num}/{total_chunks} translated successfully")

            except Exception as e:
                logger.error(f"Failed to translate chunk {chunk_num}/{total_chunks}: {str(e)}")
                # Re-raise with more context
                raise RuntimeError(
                    f"Translation failed at chunk {chunk_num}/{total_chunks}: {str(e)}"
                ) from e

        if progress_callback:
            progress_callback(total_chunks, total_chunks, "Translation complete")

        # Reassemble translated chunks
        result = self._reassemble_chunks(translated_chunks)

        logger.info(
            f"Translation complete: {len(text)} chars -> {len(result)} chars, "
            f"{total_chunks} chunk(s) processed"
        )

        return result

    def _clean_translation(self, text: str) -> str:
        """
        Clean up translation by removing overlap markers and artifacts.

        Args:
            text: The translated text to clean

        Returns:
            Cleaned translation
        """
        # Remove overlap markers that might have been translated
        markers_to_remove = [
            "[...continued from previous chunk]",
            "[...suite du fragment précédent]",  # French
            "[...continuare din fragmentul anterior]",  # Romanian
            "[...fortsetzung vom vorherigen teil]",  # German
        ]

        cleaned = text
        for marker in markers_to_remove:
            cleaned = cleaned.replace(marker, "").replace(marker.lower(), "")

        # Remove any leading/trailing whitespace
        cleaned = cleaned.strip()

        return cleaned

    def _reassemble_chunks(self, chunks: list[str]) -> str:
        """
        Reassemble translated chunks into a single text.

        Args:
            chunks: List of translated chunks

        Returns:
            Reassembled text with proper paragraph spacing
        """
        if not chunks:
            return ""

        # Join chunks with double newlines to preserve paragraph structure
        # Remove any extra whitespace that might have been introduced
        reassembled = "\n\n".join(chunk.strip() for chunk in chunks if chunk.strip())

        return reassembled

    async def translate_with_metadata(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        context: str = "",
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> dict[str, Any]:
        """
        Translate text and return both result and metadata.

        Args:
            text: The text to translate
            source_lang: Source language code
            target_lang: Target language code
            context: Optional context for translation
            progress_callback: Optional progress callback

        Returns:
            Dictionary containing:
            - translation: The translated text
            - metadata: Dict with statistics (chunks_count, original_length, etc.)
        """
        # Count chunks before translation
        chunks = self.chunking_service.chunk_text(
            text,
            max_tokens=self.settings.translation_max_tokens,
            preserve_paragraphs=True,
        )

        original_tokens = self.chunking_service.count_tokens(text)

        # Perform translation
        translation = await self.translate(
            text=text,
            source_lang=source_lang,
            target_lang=target_lang,
            context=context,
            progress_callback=progress_callback,
        )

        translated_tokens = self.chunking_service.count_tokens(translation)

        metadata = {
            "chunks_count": len(chunks),
            "original_length": len(text),
            "translated_length": len(translation),
            "original_tokens": original_tokens,
            "translated_tokens": translated_tokens,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "provider": str(self.provider),
        }

        return {"translation": translation, "metadata": metadata}

    async def validate_provider(self) -> bool:
        """
        Validate that the configured LLM provider is working.

        Returns:
            True if provider is valid and API key works, False otherwise
        """
        try:
            return await self.provider.validate_api_key()
        except Exception as e:
            logger.error(f"Provider validation failed: {str(e)}")
            return False


def get_translation_service(provider_name: str = "anthropic") -> TranslationService:
    """
    Factory function to get a translation service with specified provider.

    Args:
        provider_name: Name of the LLM provider ("anthropic", "openai", etc.)

    Returns:
        TranslationService instance with specified provider

    Raises:
        ValueError: If provider name is not supported
    """
    providers = {
        "anthropic": AnthropicProvider,
        # Future providers can be added here:
        # "openai": OpenAIProvider,
        # "gemini": GeminiProvider,
    }

    if provider_name not in providers:
        raise ValueError(
            f"Unsupported provider: {provider_name}. "
            f"Available providers: {', '.join(providers.keys())}"
        )

    provider_class = providers[provider_name]
    provider = provider_class()

    return TranslationService(provider=provider)
