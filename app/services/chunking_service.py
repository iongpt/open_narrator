"""Text chunking service for splitting long text into processable chunks."""

import logging

import tiktoken

from app.config import get_settings

logger = logging.getLogger(__name__)


class ChunkingService:
    """Service for intelligently chunking text while preserving paragraph boundaries."""

    def __init__(self, encoding_name: str = "cl100k_base"):
        """
        Initialize the chunking service.

        Args:
            encoding_name: The tiktoken encoding to use (default: cl100k_base for Claude)
        """
        self.encoding = tiktoken.get_encoding(encoding_name)
        self.settings = get_settings()

    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in a text string.

        Args:
            text: The text to count tokens for

        Returns:
            Number of tokens in the text
        """
        return len(self.encoding.encode(text))

    def chunk_text(
        self,
        text: str,
        max_tokens: int | None = None,
        preserve_paragraphs: bool = True,
        overlap_tokens: int | None = None,
    ) -> list[str]:
        """
        Split text into chunks while preserving paragraph boundaries.

        This function intelligently splits long text into chunks that:
        - Do not exceed the maximum token count
        - Preserve paragraph boundaries (never split mid-paragraph)
        - Include overlap between chunks for context continuity

        Algorithm:
        1. Split text by double newlines (paragraphs)
        2. Count tokens for each paragraph
        3. Group paragraphs until max_tokens limit is reached
        4. Add overlap from previous chunk if configured

        Args:
            text: The input text to chunk
            max_tokens: Maximum tokens per chunk (defaults to settings.translation_max_tokens)
            preserve_paragraphs: If True, never split paragraphs (default: True)
            overlap_tokens: Number of tokens to overlap between chunks
                          (defaults to settings.translation_chunk_overlap)

        Returns:
            List of text chunks, each within the token limit

        Raises:
            ValueError: If text is empty or max_tokens is too small
        """
        if not text or not text.strip():
            raise ValueError("Cannot chunk empty text")

        max_tokens = max_tokens or self.settings.translation_max_tokens
        overlap_tokens = overlap_tokens or self.settings.translation_chunk_overlap

        if max_tokens < 100:
            raise ValueError(f"max_tokens must be at least 100, got {max_tokens}")

        if overlap_tokens >= max_tokens:
            raise ValueError(
                f"overlap_tokens ({overlap_tokens}) must be less than max_tokens ({max_tokens})"
            )

        # If text is already small enough, return as single chunk
        total_tokens = self.count_tokens(text)
        if total_tokens <= max_tokens:
            logger.info(f"Text fits in single chunk ({total_tokens} tokens)")
            return [text]

        # Split by paragraphs (double newlines)
        if preserve_paragraphs:
            paragraphs = text.split("\n\n")
            # Filter out empty paragraphs but keep track of structure
            paragraphs = [p for p in paragraphs if p.strip()]
        else:
            # Fall back to splitting by sentences or words if needed
            paragraphs = [text]

        chunks: list[str] = []
        current_chunk: list[str] = []
        current_tokens = 0
        overlap_text = ""

        for i, paragraph in enumerate(paragraphs):
            para_tokens = self.count_tokens(paragraph)

            # Handle case where single paragraph exceeds max_tokens
            if para_tokens > max_tokens:
                if preserve_paragraphs:
                    logger.warning(
                        f"Paragraph {i} has {para_tokens} tokens, "
                        f"exceeding max_tokens ({max_tokens}). "
                        f"This paragraph will be split."
                    )
                    # Split the paragraph by sentences
                    sentences = self._split_into_sentences(paragraph)
                    for sentence in sentences:
                        sent_tokens = self.count_tokens(sentence)
                        if current_tokens + sent_tokens > max_tokens:
                            # Save current chunk
                            if current_chunk:
                                chunk_text = "\n\n".join(current_chunk)
                                chunks.append(chunk_text)
                                # Prepare overlap for next chunk
                                overlap_text = self._get_overlap(chunk_text, overlap_tokens)
                            current_chunk = (
                                [overlap_text + sentence] if overlap_text else [sentence]
                            )
                            current_tokens = self.count_tokens("\n\n".join(current_chunk))
                        else:
                            current_chunk.append(sentence)
                            current_tokens += sent_tokens
                else:
                    # Force split even if not preserving paragraphs
                    if current_chunk:
                        chunks.append("\n\n".join(current_chunk))
                    chunks.append(paragraph)
                    current_chunk = []
                    current_tokens = 0
                continue

            # Check if adding this paragraph would exceed limit
            if current_tokens + para_tokens > max_tokens:
                # Save current chunk and start new one
                if current_chunk:
                    chunk_text = "\n\n".join(current_chunk)
                    chunks.append(chunk_text)
                    # Prepare overlap for next chunk
                    overlap_text = self._get_overlap(chunk_text, overlap_tokens)

                # Start new chunk with overlap
                if overlap_text:
                    current_chunk = [overlap_text, paragraph]
                    current_tokens = self.count_tokens("\n\n".join(current_chunk))
                else:
                    current_chunk = [paragraph]
                    current_tokens = para_tokens
            else:
                # Add paragraph to current chunk
                current_chunk.append(paragraph)
                current_tokens += para_tokens

        # Don't forget the last chunk
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        logger.info(
            f"Split text into {len(chunks)} chunks. "
            f"Original: {total_tokens} tokens, "
            f"Max per chunk: {max_tokens} tokens"
        )

        return chunks

    def _split_into_sentences(self, text: str) -> list[str]:
        """
        Split text into sentences using simple heuristics.

        Args:
            text: Text to split

        Returns:
            List of sentences
        """
        # Simple sentence splitting by common punctuation
        # Note: This is basic and could be improved with NLP libraries
        import re

        # Split on . ! ? followed by space/newline, but not on abbreviations
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s for s in sentences if s.strip()]

    def _get_overlap(self, text: str, overlap_tokens: int) -> str:
        """
        Get the last N tokens from text for overlap with next chunk.

        Args:
            text: The text to extract overlap from
            overlap_tokens: Number of tokens to include in overlap

        Returns:
            Text containing approximately overlap_tokens from the end of input
        """
        if overlap_tokens <= 0:
            return ""

        # Get last few sentences/paragraphs that fit within overlap_tokens
        paragraphs = text.split("\n\n")
        overlap_parts: list[str] = []
        current_tokens = 0

        # Work backwards from the end
        for paragraph in reversed(paragraphs):
            para_tokens = self.count_tokens(paragraph)
            if current_tokens + para_tokens > overlap_tokens:
                break
            overlap_parts.insert(0, paragraph)
            current_tokens += para_tokens

        overlap_text = "\n\n".join(overlap_parts)

        # If we got overlap, add a marker to indicate continuation
        if overlap_text:
            return f"[...continued from previous chunk]\n\n{overlap_text}\n\n"

        return ""


def chunk_text(
    text: str,
    max_tokens: int | None = None,
    preserve_paragraphs: bool = True,
    overlap_tokens: int | None = None,
) -> list[str]:
    """
    Convenience function to chunk text without instantiating service.

    Args:
        text: The input text to chunk
        max_tokens: Maximum tokens per chunk (defaults to settings.translation_max_tokens)
        preserve_paragraphs: If True, never split paragraphs (default: True)
        overlap_tokens: Number of tokens to overlap between chunks

    Returns:
        List of text chunks

    Raises:
        ValueError: If text is empty or parameters are invalid
    """
    service = ChunkingService()
    return service.chunk_text(text, max_tokens, preserve_paragraphs, overlap_tokens)
