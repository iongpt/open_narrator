"""Anthropic Claude API provider for translation."""

import asyncio
import logging
from typing import Any

from anthropic import Anthropic, APIError, APIStatusError, RateLimitError

from app.config import get_settings
from app.providers.base import BaseLLMProvider

logger = logging.getLogger(__name__)


class AnthropicProvider(BaseLLMProvider):
    """
    Anthropic Claude API provider for translation.

    This provider implements translation using Claude AI models with:
    - Exponential backoff retry logic for rate limits
    - Comprehensive error handling
    - Token usage tracking and logging
    - Context-aware prompts for high-quality translation
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        max_retries: int = 3,
        initial_retry_delay: float = 1.0,
    ):
        """
        Initialize the Anthropic provider.

        Args:
            api_key: Anthropic API key (defaults to settings.anthropic_api_key)
            model: Model to use (defaults to settings.translation_model)
            max_retries: Maximum number of retry attempts for failed requests
            initial_retry_delay: Initial delay in seconds for exponential backoff
        """
        self.settings = get_settings()
        self.api_key = api_key or self.settings.anthropic_api_key
        self.model = model or self.settings.translation_model
        self.max_retries = max_retries
        self.initial_retry_delay = initial_retry_delay

        if not self.api_key:
            raise ValueError(
                "Anthropic API key is required. Set ANTHROPIC_API_KEY environment variable."
            )

        self.client = Anthropic(api_key=self.api_key)
        logger.info(f"Initialized Anthropic provider with model: {self.model}")

    async def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        context: str = "",
    ) -> str:
        """
        Translate text using Claude API with context-aware prompts.

        This method uses a carefully crafted prompt designed for audiobook/podcast
        translation that preserves:
        - Natural flow and tone for audio narration
        - Cultural nuances and idioms
        - Speaker's intent and emotion
        - Paragraph structure

        Args:
            text: The text to translate
            source_lang: Source language code (e.g., "en")
            target_lang: Target language code (e.g., "ro")
            context: Optional context about the content (e.g., "educational podcast", "mystery novel")

        Returns:
            The translated text

        Raises:
            ValueError: If text is empty or languages are invalid
            RuntimeError: If translation fails after all retries
        """
        if not text or not text.strip():
            raise ValueError("Text to translate cannot be empty")

        if not source_lang or not target_lang:
            raise ValueError("Both source_lang and target_lang are required")

        # Construct the context-aware prompt
        prompt = self._build_translation_prompt(text, source_lang, target_lang, context)

        # Execute translation with retry logic
        for attempt in range(self.max_retries):
            try:
                logger.info(
                    f"Translation attempt {attempt + 1}/{self.max_retries}: "
                    f"{source_lang} -> {target_lang}, {len(text)} chars"
                )

                # Call Claude API
                response = await asyncio.to_thread(
                    self.client.messages.create,
                    model=self.model,
                    max_tokens=4096,
                    messages=[{"role": "user", "content": prompt}],
                )

                # Extract translated text
                if not response.content:
                    raise RuntimeError("Empty response from Claude API")

                translated_text = str(response.content[0].text)

                # Log token usage
                input_tokens = response.usage.input_tokens
                output_tokens = response.usage.output_tokens
                logger.info(
                    f"Translation successful. Tokens - Input: {input_tokens}, Output: {output_tokens}"
                )

                return translated_text

            except RateLimitError as e:
                # Handle rate limiting with exponential backoff
                if attempt < self.max_retries - 1:
                    delay = self.initial_retry_delay * (2**attempt)
                    logger.warning(
                        f"Rate limit hit. Retrying in {delay:.1f}s... (attempt {attempt + 1}/{self.max_retries})"
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error("Rate limit exceeded after all retries")
                    raise RuntimeError(
                        f"Translation failed due to rate limiting after {self.max_retries} attempts"
                    ) from e

            except APIStatusError as e:
                # Handle API errors (4xx, 5xx)
                logger.error(f"API status error: {e.status_code} - {e.message}")
                if attempt < self.max_retries - 1 and e.status_code >= 500:
                    # Retry on server errors
                    delay = self.initial_retry_delay * (2**attempt)
                    logger.warning(f"Server error. Retrying in {delay:.1f}s...")
                    await asyncio.sleep(delay)
                else:
                    raise RuntimeError(f"Translation failed: {e.message}") from e

            except APIError as e:
                # Handle other API errors
                logger.error(f"API error: {str(e)}")
                if attempt < self.max_retries - 1:
                    delay = self.initial_retry_delay * (2**attempt)
                    logger.warning(f"API error. Retrying in {delay:.1f}s...")
                    await asyncio.sleep(delay)
                else:
                    raise RuntimeError(f"Translation failed: {str(e)}") from e

            except Exception as e:
                # Handle unexpected errors
                logger.error(f"Unexpected error during translation: {str(e)}")
                raise RuntimeError(f"Translation failed: {str(e)}") from e

        raise RuntimeError(f"Translation failed after {self.max_retries} attempts")

    def _build_translation_prompt(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        context: str,
    ) -> str:
        """
        Build a context-aware translation prompt optimized for audio narration.

        Args:
            text: The text to translate
            source_lang: Source language code
            target_lang: Target language code
            context: User-provided context

        Returns:
            The complete prompt string
        """
        context_note = f"\nAdditional context: {context}" if context else ""

        prompt = f"""You are a professional translator specializing in audiobook and podcast narration.

**Translation Task:**
- Source language: {source_lang}
- Target language: {target_lang}{context_note}

**Critical Requirements:**
1. Preserve the natural flow and tone suitable for audio narration
2. Maintain cultural nuances and adapt idioms appropriately
3. Keep the speaker's intent and emotional tone
4. Preserve paragraph structure and formatting
5. Use natural, spoken language that sounds good when read aloud

**Instructions:**
- Translate ONLY the text provided, with no explanations or commentary
- If you encounter names, keep them unless they have standard translations
- For cultural references that don't translate directly, adapt them naturally
- Maintain the same level of formality as the original

**Text to translate:**

{text}"""

        return prompt

    def get_model_info(self) -> dict[str, Any]:
        """
        Get information about the Claude model being used.

        Returns:
            Dictionary with model information
        """
        return {
            "name": self.model,
            "provider": "anthropic",
            "max_tokens": 200000,  # Claude 3.5 Sonnet context window
            "supports_streaming": False,  # Not implemented yet
        }

    async def validate_api_key(self) -> bool:
        """
        Validate the API key by making a minimal API call.

        Returns:
            True if API key is valid, False otherwise

        Raises:
            RuntimeError: If validation fails due to network errors
        """
        try:
            # Make a minimal API call to test the key
            await asyncio.to_thread(
                self.client.messages.create,
                model=self.model,
                max_tokens=10,
                messages=[{"role": "user", "content": "Hello"}],
            )
            logger.info("API key validation successful")
            return True

        except APIStatusError as e:
            if e.status_code == 401:
                logger.error("Invalid API key")
                return False
            logger.error(f"API validation error: {e.status_code} - {e.message}")
            raise RuntimeError(f"API key validation failed: {e.message}") from e

        except Exception as e:
            logger.error(f"Unexpected error during API key validation: {str(e)}")
            raise RuntimeError(f"API key validation failed: {str(e)}") from e
