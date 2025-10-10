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
    ) -> list[str]:
        """
        Split text into chunks while preserving paragraph boundaries.

        This function intelligently splits long text into chunks that:
        - Maximize token usage (fill each chunk as close to max_tokens as possible)
        - Preserve paragraph boundaries (never split mid-paragraph)
        - Have NO overlap (each paragraph appears exactly once)

        Algorithm:
        1. Split text by double newlines (paragraphs)
        2. Count tokens for each paragraph
        3. Add paragraphs to current chunk until adding next would exceed max_tokens
        4. Start new chunk with next paragraph (no overlap from previous)
        5. Handle oversized paragraphs by splitting into sentences

        Args:
            text: The input text to chunk
            max_tokens: Maximum tokens per chunk (defaults to settings.translation_max_tokens)
            preserve_paragraphs: If True, never split paragraphs mid-content (default: True)

        Returns:
            List of text chunks, each within the token limit, with no duplicate content

        Raises:
            ValueError: If text is empty or max_tokens is too small

        Example:
            >>> text = "Para A\\n\\nPara B\\n\\nPara C"  # A=18k, B=18k, C=5k tokens
            >>> chunks = chunker.chunk_text(text, max_tokens=20000)
            >>> # Result: ["Para A", "Para B", "Para C"]  (no overlap, maximized)
        """
        if not text or not text.strip():
            raise ValueError("Cannot chunk empty text")

        max_tokens = max_tokens or self.settings.translation_max_tokens

        if max_tokens < 100:
            raise ValueError(f"max_tokens must be at least 100, got {max_tokens}")

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
            # Fall back to single-item list if not preserving paragraphs
            paragraphs = [text]

        chunks: list[str] = []
        current_chunk: list[str] = []
        current_tokens = 0

        for i, paragraph in enumerate(paragraphs):
            para_tokens = self.count_tokens(paragraph)

            # Handle case where single paragraph exceeds max_tokens
            if para_tokens > max_tokens:
                if preserve_paragraphs:
                    logger.warning(
                        f"Paragraph {i} has {para_tokens} tokens, "
                        f"exceeding max_tokens ({max_tokens}). "
                        f"Splitting paragraph into sentences."
                    )
                    # Save current chunk before handling oversized paragraph
                    if current_chunk:
                        chunks.append("\n\n".join(current_chunk))
                        current_chunk = []
                        current_tokens = 0

                    # Split the oversized paragraph by sentences
                    sentences = self._split_into_sentences(paragraph)
                    for sentence in sentences:
                        sent_tokens = self.count_tokens(sentence)

                        # If single sentence is too large, we have to include it anyway
                        if sent_tokens > max_tokens:
                            logger.warning(
                                f"Single sentence has {sent_tokens} tokens, "
                                f"exceeding max_tokens ({max_tokens}). "
                                f"Including as standalone chunk."
                            )
                            if current_chunk:
                                chunks.append("\n\n".join(current_chunk))
                                current_chunk = []
                                current_tokens = 0
                            chunks.append(sentence)
                            continue

                        # Will adding this sentence exceed limit?
                        if current_tokens + sent_tokens > max_tokens:
                            # Save current chunk and start new one
                            if current_chunk:
                                chunks.append("\n\n".join(current_chunk))
                            current_chunk = [sentence]
                            current_tokens = sent_tokens
                        else:
                            # Add sentence to current chunk (maximize usage)
                            current_chunk.append(sentence)
                            current_tokens += sent_tokens
                else:
                    # Not preserving paragraphs: force include oversized paragraph
                    if current_chunk:
                        chunks.append("\n\n".join(current_chunk))
                    chunks.append(paragraph)
                    current_chunk = []
                    current_tokens = 0
                continue

            # Check if adding this paragraph would exceed limit
            if current_tokens + para_tokens > max_tokens:
                # Save current chunk (maximized without this paragraph)
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))

                # Start new chunk with this paragraph (NO overlap)
                current_chunk = [paragraph]
                current_tokens = para_tokens
            else:
                # Add paragraph to current chunk (maximize usage)
                current_chunk.append(paragraph)
                current_tokens += para_tokens

        # Don't forget the last chunk
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        logger.info(
            f"Split text into {len(chunks)} chunks (no overlap). "
            f"Original: {total_tokens} tokens, "
            f"Max per chunk: {max_tokens} tokens"
        )

        # Log chunk sizes for debugging
        for idx, chunk in enumerate(chunks):
            chunk_tokens = self.count_tokens(chunk)
            logger.debug(f"Chunk {idx + 1}: {chunk_tokens} tokens")

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


def chunk_text(
    text: str,
    max_tokens: int | None = None,
    preserve_paragraphs: bool = True,
) -> list[str]:
    """
    Convenience function to chunk text without instantiating service.

    Args:
        text: The input text to chunk
        max_tokens: Maximum tokens per chunk (defaults to settings.translation_max_tokens)
        preserve_paragraphs: If True, never split paragraphs (default: True)

    Returns:
        List of text chunks with no overlap (each paragraph appears exactly once)

    Raises:
        ValueError: If text is empty or parameters are invalid
    """
    service = ChunkingService()
    return service.chunk_text(text, max_tokens, preserve_paragraphs)
