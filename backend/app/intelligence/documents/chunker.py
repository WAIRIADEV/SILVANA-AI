
from collections.abc import Callable
from uuid import UUID, uuid4

from backend.app.intelligence.documents.extractor import (
    ExtractionResult,
)
from backend.app.intelligence.documents.models import DocumentChunk


class DocumentChunker:
    """Split extracted text into overlapping, traceable chunks."""

    DEFAULT_MAX_TOKENS = 450
    DEFAULT_TOKEN_OVERLAP = 60

    BOUNDARIES = (
        "\n\n",
        "\n",
        ". ",
        "? ",
        "! ",
        "; ",
        ", ",
        " ",
    )

    def __init__(
        self,
        *,
        chunk_size: int = 800,
        overlap: int = 120,
        min_chunk_size: int = 100,
        token_counter: Callable[[str], int] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        token_overlap: int = DEFAULT_TOKEN_OVERLAP,
    ) -> None:
        if chunk_size < 1:
            raise ValueError("Chunk size must be at least 1.")

        if overlap < 0 or overlap >= chunk_size:
            raise ValueError(
                "Overlap must be non-negative and smaller than chunk size."
            )

        if min_chunk_size < 1 or min_chunk_size > chunk_size:
            raise ValueError(
                "Minimum chunk size must be between 1 and chunk size."
            )

        if max_tokens < 3:
            raise ValueError("Maximum tokens must be at least 3.")

        if token_overlap < 0 or token_overlap >= max_tokens:
            raise ValueError(
                "Token overlap must be non-negative and smaller "
                "than the maximum token budget."
            )

        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size

        self.token_counter = token_counter
        self.max_tokens = max_tokens
        self.token_overlap = token_overlap

    def chunk(
        self,
        extraction: ExtractionResult,
        document_id: UUID,
    ) -> list[DocumentChunk]:
        """Convert extracted sections into indexed document chunks."""

        if not extraction.sections:
            return []

        groups = self._group_sections(extraction)
        chunks: list[DocumentChunk] = []

        for content, page_number in groups:
            pieces = self._split_text(content)

            for piece in pieces:
                if not piece.strip():
                    continue

                chunks.append(
                    DocumentChunk(
                        id=uuid4(),
                        document_id=document_id,
                        chunk_index=len(chunks),
                        content=piece,
                        page_number=page_number,
                    )
                )

        return chunks

    @staticmethod
    def _group_sections(
        extraction: ExtractionResult,
    ) -> list[tuple[str, int | None]]:
        """Combine related sections without losing PDF page references."""

        if extraction.format.lower() != "pdf":
            content = "\n\n".join(
                section.content.strip()
                for section in extraction.sections
                if section.content.strip()
            )

            return [(content, None)] if content else []

        groups: list[tuple[str, int | None]] = []

        for section in extraction.sections:
            content = section.content.strip()

            if not content:
                continue

            page_number = section.page_number

            if groups and groups[-1][1] == page_number:
                previous_content, _ = groups[-1]
                groups[-1] = (
                    f"{previous_content}\n\n{content}",
                    page_number,
                )
            else:
                groups.append((content, page_number))

        return groups

    def _split_text(self, content: str) -> list[str]:
        """Select character-based or token-aware splitting."""

        text = content.strip()

        if not text:
            return []

        if self.token_counter is not None:
            return self._split_text_by_tokens(text)

        return self._split_text_by_characters(text)

    def _split_text_by_characters(self, text: str) -> list[str]:
        """Preserve the original character-based chunking behavior."""

        if len(text) <= self.chunk_size:
            return [text]

        chunks: list[str] = []
        start = 0

        while start < len(text):
            hard_end = min(
                start + self.chunk_size,
                len(text),
            )

            if hard_end == len(text):
                piece = text[start:hard_end].strip()

                if piece:
                    chunks.append(piece)

                break

            search_start = min(
                start + self.min_chunk_size,
                hard_end,
            )

            boundary_end = self._find_natural_boundary(
                text,
                search_start,
                hard_end,
            )

            piece = text[start:boundary_end].strip()

            if piece:
                chunks.append(piece)

            next_start = boundary_end - self.overlap

            if next_start <= start:
                next_start = boundary_end

            start = next_start

        return chunks

    def _split_text_by_tokens(self, text: str) -> list[str]:
        """Split text using the injected model tokenizer."""

        chunks: list[str] = []
        start = 0

        while start < len(text):
            if self._count_tokens(text[start:]) <= self.max_tokens:
                piece = text[start:].strip()

                if piece:
                    chunks.append(piece)

                break

            hard_end = self._find_max_token_end(text, start)

            if hard_end <= start:
                raise ValueError(
                    "Unable to create a chunk within the token budget."
                )

            boundary_end = self._find_token_boundary(
                text,
                start,
                hard_end,
            )

            piece = text[start:boundary_end].strip()

            if piece:
                chunks.append(piece)

            if boundary_end >= len(text):
                break

            next_start = self._find_token_overlap_start(
                text,
                start,
                boundary_end,
            )

            if next_start <= start or next_start >= boundary_end:
                next_start = boundary_end

            start = next_start

        return chunks

    def _count_tokens(self, text: str) -> int:
        if self.token_counter is None:
            raise RuntimeError("Token counter is not configured.")

        count = self.token_counter(text)

        if isinstance(count, bool) or not isinstance(count, int):
            raise TypeError("Token counter must return an integer.")

        if count < 0:
            raise ValueError("Token count cannot be negative.")

        return count

    def _find_max_token_end(self, text: str, start: int) -> int:
        """Find the furthest character position within the token budget."""

        low = start + 1
        high = len(text)
        best = start

        while low <= high:
            middle = (low + high) // 2
            candidate = text[start:middle]

            if self._count_tokens(candidate) <= self.max_tokens:
                best = middle
                low = middle + 1
            else:
                high = middle - 1

        return best

    def _find_token_boundary(
        self,
        text: str,
        start: int,
        hard_end: int,
    ) -> int:
        """Prefer a natural boundary without exceeding the token budget."""

        minimum_end = min(
            start + self.min_chunk_size,
            hard_end,
        )

        for boundary in self.BOUNDARIES:
            position = text.rfind(
                boundary,
                minimum_end,
                hard_end,
            )

            if position != -1:
                candidate_end = position + len(boundary)

                if (
                    candidate_end > start
                    and self._count_tokens(
                        text[start:candidate_end]
                    ) <= self.max_tokens
                ):
                    return candidate_end

        return hard_end

    def _find_token_overlap_start(
        self,
        text: str,
        start: int,
        end: int,
    ) -> int:
        """Find a starting position with the configured token overlap."""

        if self.token_overlap == 0:
            return end

        low = start + 1
        high = end
        best = end

        while low <= high:
            middle = (low + high) // 2
            candidate = text[middle:end]

            if self._count_tokens(candidate) <= self.token_overlap:
                best = middle
                high = middle - 1
            else:
                low = middle + 1

        return best

    def _find_natural_boundary(
        self,
        text: str,
        search_start: int,
        hard_end: int,
    ) -> int:
        for boundary in self.BOUNDARIES:
            position = text.rfind(
                boundary,
                search_start,
                hard_end,
            )

            if position != -1:
                candidate_end = position + len(boundary)

                if candidate_end > search_start:
                    return candidate_end

        return hard_end
