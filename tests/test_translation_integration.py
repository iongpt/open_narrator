"""Integration tests for translation pipeline to verify no text duplication."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.services.chunking_service import ChunkingService
from app.services.translation_service import TranslationService


class TestTranslationIntegration:
    """Integration tests for translation with zero-overlap chunking."""

    @pytest.fixture
    def mock_llm_provider(self) -> MagicMock:
        """Create a mock LLM provider that echoes input with a prefix."""
        provider = MagicMock()

        # Mock translate method to echo input with language prefix
        async def mock_translate(
            text: str, source_lang: str, target_lang: str, context: str = ""
        ) -> str:
            # Simple echo translation for testing
            return f"[TRANSLATED:{target_lang}] {text}"

        provider.translate = AsyncMock(side_effect=mock_translate)
        return provider

    @pytest.fixture
    def translation_service(self, mock_llm_provider: MagicMock) -> TranslationService:
        """Create translation service with mocked provider."""
        return TranslationService(provider=mock_llm_provider, chunking_service=ChunkingService())

    @pytest.mark.asyncio
    async def test_no_text_duplication_in_translation(
        self, translation_service: TranslationService
    ) -> None:
        """Critical test: Verify no text duplication in translated output."""
        # Create text with unique, identifiable paragraphs with longer content to force chunking
        # Use padded markers to avoid substring matching (e.g., "_001_" vs "_010_")
        paragraphs = [
            f"This is paragraph with unique marker id_PARA{i:04d}_END. "
            f"Adding more text here to make the paragraph longer so we can test chunking properly."
            for i in range(30)
        ]
        original_text = "\n\n".join(paragraphs)

        # Translate with small max_tokens to force multiple chunks
        translation_service.chunking_service.settings.translation_max_tokens = 300

        result = await translation_service.translate(
            text=original_text, source_lang="en", target_lang="ro", context="test"
        )

        # Verify each paragraph appears exactly once in translation
        for i in range(len(paragraphs)):
            # Use unique marker with boundaries that won't substring match
            unique_marker = f"id_PARA{i:04d}_END"
            count = result.count(unique_marker)
            assert (
                count == 1
            ), f"Paragraph {i} (marker {unique_marker}) appears {count} times in translation (expected 1)"

    @pytest.mark.asyncio
    async def test_no_repeated_sentences_in_output(
        self, translation_service: TranslationService
    ) -> None:
        """Test that no sentences are repeated in the translated output."""
        # Create text with highly distinct sentences
        sentences_per_para = 5
        paragraphs = []
        for i in range(10):
            para_sentences = [
                f"Sentence {i}-{j} has unique marker {i*1000+j}." for j in range(sentences_per_para)
            ]
            paragraphs.append(" ".join(para_sentences))

        original_text = "\n\n".join(paragraphs)

        # Force chunking
        translation_service.chunking_service.settings.translation_max_tokens = 150

        result = await translation_service.translate(
            text=original_text, source_lang="en", target_lang="ro"
        )

        # Extract all unique markers from result
        import re

        markers = re.findall(r"unique marker (\d+)", result)

        # Verify no marker appears twice
        unique_markers = set(markers)
        assert len(markers) == len(unique_markers), (
            f"Found duplicate markers in translation: "
            f"{len(markers)} total vs {len(unique_markers)} unique"
        )

    @pytest.mark.asyncio
    async def test_reassembled_chunks_preserve_order(
        self, translation_service: TranslationService
    ) -> None:
        """Test that chunk reassembly preserves original paragraph order."""
        # Create ordered paragraphs with sequence numbers
        paragraphs = [f"Paragraph sequence {i:03d}." for i in range(30)]
        original_text = "\n\n".join(paragraphs)

        # Force multiple chunks
        translation_service.chunking_service.settings.translation_max_tokens = 100

        result = await translation_service.translate(
            text=original_text, source_lang="en", target_lang="ro"
        )

        # Extract sequence numbers from result
        import re

        sequences = re.findall(r"sequence (\d+)", result)
        sequences = [int(s) for s in sequences]

        # Verify sequences are in order and complete
        expected = list(range(30))
        assert sequences == expected, f"Sequence order mismatch: {sequences} != {expected}"

    @pytest.mark.asyncio
    async def test_large_text_no_duplication(self, translation_service: TranslationService) -> None:
        """Test large text (simulating audiobook) has no duplication."""
        # Simulate a large transcript with 500 paragraphs
        paragraphs = [
            f"Audiobook chapter content paragraph {i} with story details." for i in range(500)
        ]
        original_text = "\n\n".join(paragraphs)

        # Use realistic max_tokens
        translation_service.chunking_service.settings.translation_max_tokens = 1000

        result = await translation_service.translate(
            text=original_text, source_lang="en", target_lang="ro"
        )

        # Verify each paragraph number appears exactly once
        # Use word boundaries to avoid substring matches
        import re

        for i in range(500):
            # Use regex with word boundaries to match exact paragraph numbers
            pattern = rf"\bparagraph {i}\b"
            matches = re.findall(pattern, result)
            count = len(matches)
            assert count == 1, f"Paragraph {i} appears {count} times (expected 1)"

    @pytest.mark.asyncio
    async def test_token_count_preservation(self, translation_service: TranslationService) -> None:
        """Test that total token count is preserved (no loss, no duplication)."""
        # Create text with known structure
        paragraphs = [f"Test paragraph {i} content." for i in range(50)]
        original_text = "\n\n".join(paragraphs)

        # Force chunking
        translation_service.chunking_service.settings.translation_max_tokens = 200

        result = await translation_service.translate(
            text=original_text, source_lang="en", target_lang="ro"
        )

        # In this test, we're checking that each paragraph appears once
        import re

        for i in range(50):
            # Use word boundaries to avoid substring matches
            pattern = rf"\bparagraph {i}\b"
            matches = re.findall(pattern, result)
            count = len(matches)
            assert count == 1, f"Token preservation check failed for paragraph {i}"

    @pytest.mark.asyncio
    async def test_progress_callback_invoked_correctly(
        self, translation_service: TranslationService
    ) -> None:
        """Test that progress callback is called correctly during chunked translation."""
        progress_calls: list[dict[str, Any]] = []

        def progress_callback(current: int, total: int, message: str) -> None:
            progress_calls.append({"current": current, "total": total, "message": message})

        # Create text that will be split into ~3 chunks
        paragraphs = [f"Paragraph {i} with content." for i in range(30)]
        original_text = "\n\n".join(paragraphs)

        translation_service.chunking_service.settings.translation_max_tokens = 150

        await translation_service.translate(
            text=original_text,
            source_lang="en",
            target_lang="ro",
            progress_callback=progress_callback,
        )

        # Verify progress callback was called
        assert len(progress_calls) > 0, "Progress callback was never called"

        # Verify final progress indicates completion
        final_call = progress_calls[-1]
        assert final_call["current"] == final_call["total"], "Final progress not 100%"
        assert (
            "complete" in final_call["message"].lower()
        ), "Final message doesn't indicate completion"

    @pytest.mark.asyncio
    async def test_empty_paragraphs_handled_correctly(
        self, translation_service: TranslationService
    ) -> None:
        """Test that empty paragraphs don't cause duplication issues."""
        # Mix of content and empty paragraphs
        paragraphs = [
            "Paragraph 1.",
            "",  # Empty
            "Paragraph 2.",
            "   ",  # Whitespace only
            "Paragraph 3.",
        ]
        original_text = "\n\n".join(paragraphs)

        result = await translation_service.translate(
            text=original_text, source_lang="en", target_lang="ro"
        )

        # Verify each non-empty paragraph appears exactly once
        assert result.count("Paragraph 1") == 1
        assert result.count("Paragraph 2") == 1
        assert result.count("Paragraph 3") == 1

    @pytest.mark.asyncio
    async def test_with_metadata_no_duplication(
        self, translation_service: TranslationService
    ) -> None:
        """Test translate_with_metadata doesn't introduce duplication."""
        # Create longer paragraphs to force chunking
        paragraphs = [
            f"Content paragraph number {i} with additional text to make it longer."
            for i in range(30)
        ]
        original_text = "\n\n".join(paragraphs)

        translation_service.chunking_service.settings.translation_max_tokens = 200

        result = await translation_service.translate_with_metadata(
            text=original_text, source_lang="en", target_lang="ro"
        )

        translation = result["translation"]
        metadata = result["metadata"]

        # Verify no duplication using word boundaries
        import re

        for i in range(30):
            # Use word boundaries to avoid substring matches
            pattern = rf"\bparagraph number {i}\b"
            matches = re.findall(pattern, translation)
            count = len(matches)
            assert count == 1, f"Content paragraph {i} appears {count} times"

        # Verify metadata is present
        assert "chunks_count" in metadata
        assert metadata["chunks_count"] >= 1  # At least 1 chunk
        assert "original_tokens" in metadata
        assert "translated_tokens" in metadata
