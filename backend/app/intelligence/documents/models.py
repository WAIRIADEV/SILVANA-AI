
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from uuid import UUID


class DocumentScope(StrEnum):
    """Defines which conversations can access a document."""

    TEMPORARY = "temporary"
    PERSISTENT = "persistent"


class DocumentStatus(StrEnum):
    """Tracks a document through its processing lifecycle."""

    UPLOADED = "uploaded"
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class DocumentRecord:
    """Metadata describing a stored document."""

    id: UUID
    scope: DocumentScope
    conversation_id: UUID | None

    filename: str
    storage_key: str
    content_hash: str
    mime_type: str
    size_bytes: int

    status: DocumentStatus = DocumentStatus.UPLOADED

    page_count: int | None = None
    chunk_count: int = 0
    error_message: str | None = None

    created_at: datetime | None = None
    last_activity_at: datetime | None = None
    expires_at: datetime | None = None
    indexed_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class DocumentChunk:
    """An extracted section of a document."""

    id: UUID
    document_id: UUID

    chunk_index: int
    content: str
    page_number: int | None = None

    created_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class DocumentEmbedding:
    """A vector representation of an indexed document chunk."""

    chunk_id: UUID
    model_name: str
    dimensions: int
    vector: list[float]

    created_at: datetime | None = None
