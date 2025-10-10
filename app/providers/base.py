"""Abstract base class for LLM translation providers."""

from abc import ABC, abstractmethod
from typing import Any


class BaseLLMProvider(ABC):
    """
    Abstract base class for LLM translation providers.

    This class defines the interface that all LLM providers must implement.
    It allows easy addition of new providers (OpenAI, Gemini, etc.) without
    changing the translation service code.

    Usage:
        class MyProvider(BaseLLMProvider):
            async def translate(self, text, source_lang, target_lang, context):
                # Implementation
                return translated_text
    """

    @abstractmethod
    async def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        context: str = "",
    ) -> str:
        """
        Translate text from source language to target language.

        Args:
            text: The text to translate
            source_lang: Source language code (e.g., "en", "ro")
            target_lang: Target language code (e.g., "en", "ro")
            context: Optional context for the translation (e.g., "formal email", "audiobook narration")

        Returns:
            The translated text

        Raises:
            ValueError: If parameters are invalid
            RuntimeError: If translation fails
        """
        pass

    @abstractmethod
    def get_model_info(self) -> dict[str, Any]:
        """
        Get information about the model being used.

        Returns:
            Dictionary containing model information:
            - name: Model name/identifier
            - provider: Provider name (e.g., "anthropic", "openai")
            - max_tokens: Maximum tokens supported
            - supports_streaming: Whether streaming is supported
        """
        pass

    @abstractmethod
    async def validate_api_key(self) -> bool:
        """
        Validate that the API key is valid and working.

        Returns:
            True if API key is valid, False otherwise

        Raises:
            RuntimeError: If validation fails due to network or other errors
        """
        pass

    def __str__(self) -> str:
        """String representation of the provider."""
        info = self.get_model_info()
        return f"{info.get('provider', 'Unknown')} ({info.get('name', 'Unknown')})"
