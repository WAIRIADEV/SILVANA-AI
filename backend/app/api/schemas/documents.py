
from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from backend.app.intelligence.documents.models import (
    DocumentScope,
    DocumentStatus,
)


class DocumentResponse(BaseModel):
    """Safe public representation of a stored document."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    scope: DocumentScope
    conversation_id: UUID | None

    filename: str
    content_hash: str
    mime_type: str
    size_bytes: int

    status: DocumentStatus

    page_count: int | None = None
    chunk_count: int = 0

    created_at: datetime | None = None
    last_activity_at: datetime | None = None
    expires_at: datetime | None = None
    indexed_at: datetime | None = None
