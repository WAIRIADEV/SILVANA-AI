from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

from backend.app.intelligence.documents.models import DocumentScope
from backend.app.intelligence.documents.repository import DocumentRepository


class QueryEmbeddingProvider(Protocol):
    """Interface required by the document retrieval service."""

    model_name: str
    dimensions: int

    def embed_query(self, text: str) -> list[float]:
        ...


@dataclass(frozen=True, slots=True)
class DocumentRetrievalResult:
    """A retrieved document chunk with source attribution."""

    document_id: UUID
    filename: str
    scope: DocumentScope
    chunk_id: UUID
    chunk_index: int
    content: str
    page_number: int | None
    similarity: float


class DocumentRetrievalService:
    """Coordinates query embedding and scoped document retrieval."""

    VALID_SCOPES = frozenset({
        "none",
        "conversation",
        "library",
        "both",
    })

    def __init__(
        self,
        repository: DocumentRepository,
        embedding_provider: QueryEmbeddingProvider,
    ) -> None:
        self.repository = repository
        self.embedding_provider = embedding_provider

    def retrieve(
        self,
        query: str,
        *,
        knowledge_scope: str = "none",
        conversation_id: UUID | None = None,
        limit: int = 5,
    ) -> list[DocumentRetrievalResult]:
        """Retrieve the most relevant chunks for a user's query."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Retrieval query cannot be empty.")

        if knowledge_scope not in self.VALID_SCOPES:
            raise ValueError("Unsupported knowledge scope.")

        if limit <= 0:
            raise ValueError("Retrieval limit must be greater than zero.")

        if (
            knowledge_scope in {"conversation", "both"}
            and conversation_id is None
        ):
            raise ValueError(
                "A conversation ID is required for this knowledge scope."
            )

        if knowledge_scope == "none":
            return []

        query_vector = self.embedding_provider.embed_query(query.strip())

        matches = self.repository.search_chunks(
            query_vector,
            model_name=self.embedding_provider.model_name,
            dimensions=self.embedding_provider.dimensions,
            knowledge_scope=knowledge_scope,
            conversation_id=conversation_id,
            limit=limit,
        )

        return [
            DocumentRetrievalResult(
                document_id=match["document_id"],
                filename=str(match["filename"]),
                scope=match["scope"],
                chunk_id=match["chunk_id"],
                chunk_index=int(match["chunk_index"]),
                content=str(match["content"]),
                page_number=(
                    int(match["page_number"])
                    if match["page_number"] is not None
                    else None
                ),
                similarity=float(match["similarity"]),
            )
            for match in matches
        ]
