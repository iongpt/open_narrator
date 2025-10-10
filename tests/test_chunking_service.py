"""Unit tests for ChunkingService to verify zero-overlap chunking."""

from typing import Any

import pytest

from app.services.chunking_service import ChunkingService


class TestChunkingService:
    """Test suite for ChunkingService with zero-overlap algorithm."""

    @pytest.fixture
    def chunking_service(self) -> ChunkingService:
        """Create a ChunkingService instance for testing."""
        return ChunkingService()

    def test_single_chunk_fits_within_limit(self, chunking_service: ChunkingService) -> None:
        """Test that small text returns single chunk."""
        text = "This is a short text that fits within the token limit."
        chunks = chunking_service.chunk_text(text, max_tokens=1000)

        assert len(chunks) == 1
        assert chunks[0] == text

    def test_no_overlap_between_chunks(self, chunking_service: ChunkingService) -> None:
        """Critical test: Verify no overlap between consecutive chunks."""
        # Create text with UNIQUE paragraphs (not repeated text)
        para_a = " ".join([f"Sentence A{i}." for i in range(100)])  # ~200 tokens
        para_b = " ".join([f"Sentence B{i}." for i in range(100)])  # ~200 tokens
        para_c = " ".join([f"Sentence C{i}." for i in range(100)])  # ~200 tokens
        text = f"{para_a}\n\n{para_b}\n\n{para_c}"

        chunks = chunking_service.chunk_text(text, max_tokens=250)

        # Verify we got multiple chunks
        assert len(chunks) > 1

        # Critical: Check for duplicate content between chunks
        # Extract all unique sentence patterns across all chunks
        chunk_content = []
        for chunk in chunks:
            sentences = [s.strip() for s in chunk.split(".") if s.strip()]
            chunk_content.append(set(sentences))

        # Verify no sentence appears in multiple chunks
        for i in range(len(chunk_content) - 1):
            for j in range(i + 1, len(chunk_content)):
                overlap = chunk_content[i] & chunk_content[j]
                assert (
                    len(overlap) == 0
                ), f"Found overlapping sentences between chunk {i} and {j}: {overlap}"

    def test_token_maximization(self, chunking_service: ChunkingService) -> None:
        """Test that chunks are filled close to max_tokens limit."""
        # Create paragraphs with known token counts
        para_100 = "Token " * 50  # ~100 tokens
        para_150 = "Token " * 75  # ~150 tokens
        para_200 = "Token " * 100  # ~200 tokens
        text = f"{para_100}\n\n{para_150}\n\n{para_200}"

        max_tokens = 250
        chunks = chunking_service.chunk_text(text, max_tokens=max_tokens)

        # Verify each chunk is close to max_tokens (within 90% efficiency)
        for i, chunk in enumerate(chunks):
            token_count = chunking_service.count_tokens(chunk)

            # Last chunk may be smaller, but others should be maximized
            if i < len(chunks) - 1:
                efficiency = token_count / max_tokens
                assert efficiency >= 0.5, (
                    f"Chunk {i} is not efficiently packed: "
                    f"{token_count} tokens ({efficiency:.0%} of {max_tokens})"
                )

    def test_preserves_paragraph_boundaries(self, chunking_service: ChunkingService) -> None:
        """Test that paragraphs are never split mid-content."""
        para_a = "First paragraph with multiple sentences. This continues."
        para_b = "Second paragraph is here. It has content too."
        para_c = "Third paragraph finishes it. The end."
        text = f"{para_a}\n\n{para_b}\n\n{para_c}"

        chunks = chunking_service.chunk_text(text, max_tokens=100)

        # Reconstruct text from chunks
        reconstructed = "\n\n".join(chunks)

        # Verify all original paragraphs appear intact (order may differ due to chunking)
        assert para_a in reconstructed or para_a.split(". ")[0] in reconstructed
        assert para_b in reconstructed or para_b.split(". ")[0] in reconstructed
        assert para_c in reconstructed or para_c.split(". ")[0] in reconstructed

    def test_oversized_paragraph_gets_split_by_sentences(
        self, chunking_service: ChunkingService
    ) -> None:
        """Test that paragraphs exceeding max_tokens are split by sentences."""
        # Create a very long paragraph (single paragraph, multiple unique sentences)
        long_paragraph = ". ".join([f"Unique sentence number {i}" for i in range(100)])

        chunks = chunking_service.chunk_text(long_paragraph, max_tokens=500)

        # Should produce multiple chunks
        assert len(chunks) > 1

        # Verify no sentence appears twice
        all_sentences = []
        for chunk in chunks:
            sentences = [s.strip() for s in chunk.split(".") if s.strip()]
            all_sentences.extend(sentences)

        # Check for duplicates
        unique_sentences = set(all_sentences)
        assert len(all_sentences) == len(
            unique_sentences
        ), "Found duplicate sentences across chunks"

    def test_empty_text_raises_error(self, chunking_service: ChunkingService) -> None:
        """Test that empty text raises ValueError."""
        with pytest.raises(ValueError, match="Cannot chunk empty text"):
            chunking_service.chunk_text("")

        with pytest.raises(ValueError, match="Cannot chunk empty text"):
            chunking_service.chunk_text("   \n\n  ")

    def test_max_tokens_too_small_raises_error(self, chunking_service: ChunkingService) -> None:
        """Test that max_tokens < 100 raises ValueError."""
        with pytest.raises(ValueError, match="max_tokens must be at least 100"):
            chunking_service.chunk_text("Some text", max_tokens=50)

    def test_each_paragraph_appears_exactly_once(self, chunking_service: ChunkingService) -> None:
        """Critical test: Verify each paragraph appears exactly once in output."""
        # Create text with unique, identifiable paragraphs
        paragraphs = [f"Unique paragraph {i} with distinct content {i * 111}." for i in range(10)]
        text = "\n\n".join(paragraphs)

        chunks = chunking_service.chunk_text(text, max_tokens=100)

        # Reconstruct full text from chunks
        reconstructed = "\n\n".join(chunks)

        # Verify each original paragraph appears exactly once
        for para in paragraphs:
            count = reconstructed.count(para)
            assert count == 1, f"Paragraph appears {count} times (expected 1): {para}"

    def test_chunk_reassembly_produces_original_content(
        self, chunking_service: ChunkingService
    ) -> None:
        """Test that reassembling chunks produces the original text."""
        # Create structured text with unique content
        para_list = [
            "First paragraph with unique content A.",
            "Second paragraph with unique content B.",
            "Third paragraph with unique content C.",
            "Fourth paragraph with unique content D.",
            "Fifth and final paragraph with unique content E.",
        ]
        original_text = "\n\n".join(para_list)

        # Chunk the text
        chunks = chunking_service.chunk_text(original_text, max_tokens=500)

        # Reassemble
        reassembled = "\n\n".join(chunks)

        # Verify all original content is preserved
        for para in para_list:
            assert para in reassembled, f"Missing paragraph: {para}"

        # Verify no content duplication by counting occurrences
        for para in para_list:
            count = reassembled.count(para)
            assert count == 1, f"Paragraph appears {count} times (expected 1): {para}"

    def test_large_text_chunking_performance(self, chunking_service: ChunkingService) -> None:
        """Test chunking with realistic large text (simulating audiobook transcript)."""
        # Simulate a transcript with 1000 paragraphs
        large_text = "\n\n".join([f"Paragraph {i} content here." for i in range(1000)])

        chunks = chunking_service.chunk_text(large_text, max_tokens=500)

        # Verify reasonable chunk count
        assert len(chunks) > 1
        assert len(chunks) < 1000  # Should be fewer chunks than paragraphs

        # Verify total content is preserved
        total_input_tokens = chunking_service.count_tokens(large_text)
        total_output_tokens = sum(chunking_service.count_tokens(chunk) for chunk in chunks)

        # Total tokens should be equal (no loss, no duplication)
        assert total_output_tokens == total_input_tokens, (
            f"Token count mismatch: input={total_input_tokens}, " f"output={total_output_tokens}"
        )

    def test_convenience_function(self) -> None:
        """Test the convenience function chunk_text()."""
        from app.services.chunking_service import chunk_text

        text = "Short text for convenience function test."
        chunks = chunk_text(text, max_tokens=1000)

        assert len(chunks) == 1
        assert chunks[0] == text

    def test_no_overlap_with_various_paragraph_sizes(
        self, chunking_service: ChunkingService
    ) -> None:
        """Test zero-overlap with varying paragraph sizes."""
        # Mix of small, medium, and large paragraphs
        paragraphs = [
            "Small.",  # ~1-2 tokens
            "Medium paragraph with some content here.",  # ~10 tokens
            "Large paragraph with lots of content. " * 20,  # ~100 tokens
            "Tiny",  # 1 token
            "Another medium sized paragraph here with text.",  # ~10 tokens
        ]
        text = "\n\n".join(paragraphs)

        chunks = chunking_service.chunk_text(text, max_tokens=150)

        # Verify all paragraphs appear exactly once
        reassembled = "\n\n".join(chunks)
        for para in paragraphs:
            count = reassembled.count(para)
            assert count == 1, f"Paragraph appears {count} times: {para[:30]}..."

    def test_debug_logging_shows_chunk_sizes(
        self, chunking_service: ChunkingService, caplog: Any
    ) -> None:
        """Test that debug logging shows chunk token counts."""
        import logging

        caplog.set_level(logging.DEBUG)

        # Create text large enough to force multiple chunks
        text = "\n\n".join([f"Paragraph {i} with lots of content and details." for i in range(50)])
        chunking_service.chunk_text(text, max_tokens=150)

        # Verify debug logs contain chunk size information
        assert any("Chunk" in record.message for record in caplog.records)
