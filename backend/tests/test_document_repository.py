
import sqlite3
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from backend.app.intelligence.documents.models import (
    DocumentChunk,
    DocumentRecord,
    DocumentScope,
    DocumentStatus,
)
from backend.app.intelligence.documents.repository import (
    DocumentRepository,
)


@pytest.fixture
def repository(tmp_path):
    repo = DocumentRepository(tmp_path / "documents.db")
    repo.initialize()
    return repo


def make_document(
    *,
    scope=DocumentScope.TEMPORARY,
    conversation_id=None,
    expires_at=None,
    created_at=None,
    storage_key=None,
):
    if scope == DocumentScope.TEMPORARY:
        conversation_id = conversation_id or uuid4()
        expires_at = expires_at or (
            datetime.now(timezone.utc) + timedelta(hours=24)
        )
    else:
        conversation_id = None
        expires_at = None

    return DocumentRecord(
        id=uuid4(),
        scope=scope,
        conversation_id=conversation_id,
        filename="sample.txt",
        storage_key=storage_key or f"{uuid4()}.txt",
        content_hash="a" * 64,
        mime_type="text/plain",
        size_bytes=128,
        created_at=created_at,
        expires_at=expires_at,
    )


def test_initialize_creates_database(repository):
    assert repository.database_path.exists()


def test_create_and_get_temporary_document(repository):
    document = make_document()

    repository.create_document(document)

    retrieved = repository.get_document(document.id)

    assert retrieved is not None
    assert retrieved.id == document.id
    assert retrieved.scope == DocumentScope.TEMPORARY
    assert retrieved.conversation_id == document.conversation_id
    assert retrieved.filename == document.filename
    assert retrieved.status == DocumentStatus.UPLOADED
    assert retrieved.created_at is not None
    assert retrieved.last_activity_at is not None


def test_create_and_get_persistent_document(repository):
    document = make_document(scope=DocumentScope.PERSISTENT)

    repository.create_document(document)

    retrieved = repository.get_document(document.id)

    assert retrieved is not None
    assert retrieved.scope == DocumentScope.PERSISTENT
    assert retrieved.conversation_id is None
    assert retrieved.expires_at is None


def test_temporary_document_requires_conversation_id(repository):
    document = DocumentRecord(
        id=uuid4(),
        scope=DocumentScope.TEMPORARY,
        conversation_id=None,
        filename="sample.txt",
        storage_key=f"{uuid4()}.txt",
        content_hash="a" * 64,
        mime_type="text/plain",
        size_bytes=128,
        created_at=datetime.now(timezone.utc),
        last_activity_at=datetime.now(timezone.utc),
        expires_at=datetime.now(timezone.utc) + timedelta(hours=24),
    )

    with pytest.raises(sqlite3.IntegrityError):
        repository.create_document(document)


def test_persistent_document_cannot_have_conversation_id(repository):
    document = DocumentRecord(
        id=uuid4(),
        scope=DocumentScope.PERSISTENT,
        conversation_id=uuid4(),
        filename="sample.txt",
        storage_key=f"{uuid4()}.txt",
        content_hash="b" * 64,
        mime_type="text/plain",
        size_bytes=128,
        created_at=datetime.now(timezone.utc),
        last_activity_at=datetime.now(timezone.utc),
    )

    with pytest.raises(sqlite3.IntegrityError):
        repository.create_document(document)


def test_temporary_document_requires_expiry(repository):
    document = make_document()

    invalid_document = DocumentRecord(
        id=document.id,
        scope=document.scope,
        conversation_id=document.conversation_id,
        filename=document.filename,
        storage_key=document.storage_key,
        content_hash=document.content_hash,
        mime_type=document.mime_type,
        size_bytes=document.size_bytes,
        created_at=document.created_at,
        expires_at=None,
    )

    with pytest.raises(sqlite3.IntegrityError):
        repository.create_document(invalid_document)


def test_list_documents_filters_by_scope(repository):
    temporary = make_document()
    persistent = make_document(scope=DocumentScope.PERSISTENT)

    repository.create_document(temporary)
    repository.create_document(persistent)

    results = repository.list_documents(
        scope=DocumentScope.PERSISTENT
    )

    assert len(results) == 1
    assert results[0].id == persistent.id


def test_list_documents_filters_by_conversation(repository):
    conversation_a = uuid4()
    conversation_b = uuid4()

    document_a = make_document(conversation_id=conversation_a)
    document_b = make_document(conversation_id=conversation_b)

    repository.create_document(document_a)
    repository.create_document(document_b)

    results = repository.list_documents(
        scope=DocumentScope.TEMPORARY,
        conversation_id=conversation_a,
    )

    assert len(results) == 1
    assert results[0].id == document_a.id


def test_add_chunks_updates_count_and_preserves_order(repository):
    document = make_document()
    repository.create_document(document)

    chunks = [
        DocumentChunk(
            id=uuid4(),
            document_id=document.id,
            chunk_index=1,
            content="Second section",
            page_number=2,
        ),
        DocumentChunk(
            id=uuid4(),
            document_id=document.id,
            chunk_index=0,
            content="First section",
            page_number=1,
        ),
    ]

    repository.add_chunks(chunks)

    retrieved = repository.get_document(document.id)
    stored_chunks = repository.list_chunks(document.id)

    assert retrieved is not None
    assert retrieved.chunk_count == 2
    assert [chunk.chunk_index for chunk in stored_chunks] == [0, 1]
    assert stored_chunks[0].content == "First section"
    assert stored_chunks[1].content == "Second section"


def test_add_chunks_is_atomic(repository):
    document = make_document()
    repository.create_document(document)

    duplicate_chunks = [
        DocumentChunk(
            id=uuid4(),
            document_id=document.id,
            chunk_index=0,
            content="First version",
        ),
        DocumentChunk(
            id=uuid4(),
            document_id=document.id,
            chunk_index=0,
            content="Duplicate index",
        ),
    ]

    with pytest.raises(sqlite3.IntegrityError):
        repository.add_chunks(duplicate_chunks)

    assert repository.list_chunks(document.id) == []

    retrieved = repository.get_document(document.id)
    assert retrieved is not None
    assert retrieved.chunk_count == 0


def test_delete_document_cascades_to_chunks(repository):
    document = make_document()
    repository.create_document(document)

    repository.add_chunks([
        DocumentChunk(
            id=uuid4(),
            document_id=document.id,
            chunk_index=0,
            content="Document content",
        )
    ])

    assert repository.delete_document(document.id) is True
    assert repository.get_document(document.id) is None
    assert repository.list_chunks(document.id) == []


def test_delete_conversation_documents_preserves_other_documents(
    repository,
):
    conversation_a = uuid4()
    conversation_b = uuid4()

    document_a = make_document(conversation_id=conversation_a)
    document_b = make_document(conversation_id=conversation_b)
    persistent = make_document(scope=DocumentScope.PERSISTENT)

    for document in (document_a, document_b, persistent):
        repository.create_document(document)

    deleted_count = repository.delete_conversation_documents(
        conversation_a
    )

    assert deleted_count == 1
    assert repository.get_document(document_a.id) is None
    assert repository.get_document(document_b.id) is not None
    assert repository.get_document(persistent.id) is not None


def test_touch_conversation_documents_refreshes_expiry(repository):
    conversation_id = uuid4()

    old_expiry = datetime.now(timezone.utc) - timedelta(hours=1)

    document = make_document(
        conversation_id=conversation_id,
        expires_at=old_expiry,
    )

    repository.create_document(document)

    updated_count = repository.touch_conversation_documents(
        conversation_id,
        retention_hours=24,
    )

    refreshed = repository.get_document(document.id)

    assert updated_count == 1
    assert refreshed is not None
    assert refreshed.last_activity_at is not None
    assert refreshed.expires_at is not None
    assert refreshed.expires_at > old_expiry


def test_touch_does_not_refresh_other_conversations(repository):
    conversation_a = uuid4()
    conversation_b = uuid4()

    document_a = make_document(conversation_id=conversation_a)
    document_b = make_document(conversation_id=conversation_b)

    repository.create_document(document_a)
    repository.create_document(document_b)

    original_activity = repository.get_document(
        document_b.id
    ).last_activity_at

    repository.touch_conversation_documents(
        conversation_a,
        retention_hours=24,
    )

    unchanged = repository.get_document(document_b.id)

    assert unchanged is not None
    assert unchanged.last_activity_at == original_activity


def test_touch_rejects_invalid_retention(repository):
    with pytest.raises(ValueError):
        repository.touch_conversation_documents(
            uuid4(),
            retention_hours=0,
        )


def test_delete_expired_documents_only_removes_expired_temporary(
    repository,
):
    now = datetime.now(timezone.utc)

    expired = make_document(
        expires_at=now - timedelta(hours=1),
    )

    active = make_document(
        expires_at=now + timedelta(hours=24),
    )

    persistent = make_document(
        scope=DocumentScope.PERSISTENT,
    )

    for document in (expired, active, persistent):
        repository.create_document(document)

    deleted_count = repository.delete_expired_documents(now=now)

    assert deleted_count == 1
    assert repository.get_document(expired.id) is None
    assert repository.get_document(active.id) is not None
    assert repository.get_document(persistent.id) is not None


def test_delete_expired_documents_rejects_naive_datetime(repository):
    naive_now = datetime.now()

    with pytest.raises(ValueError):
        repository.delete_expired_documents(now=naive_now)


def test_create_document_rejects_naive_timestamp(repository):
    document = make_document(
        created_at=datetime.now(),
    )

    with pytest.raises(ValueError):
        repository.create_document(document)


def test_update_document_status(repository):
    document = make_document()
    repository.create_document(document)

    updated = repository.update_document_status(
        document.id,
        DocumentStatus.INDEXED,
        page_count=3,
        chunk_count=5,
    )

    retrieved = repository.get_document(document.id)

    assert updated is True
    assert retrieved is not None
    assert retrieved.status == DocumentStatus.INDEXED
    assert retrieved.page_count == 3
    assert retrieved.chunk_count == 5
    assert retrieved.indexed_at is not None


def test_update_missing_document_returns_false(repository):
    assert repository.update_document_status(
        uuid4(),
        DocumentStatus.FAILED,
    ) is False


def test_delete_missing_document_returns_false(repository):
    assert repository.delete_document(uuid4()) is False
