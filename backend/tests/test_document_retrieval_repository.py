from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from backend.app.intelligence.documents.models import (
    DocumentChunk,
    DocumentEmbedding,
    DocumentRecord,
    DocumentScope,
    DocumentStatus,
)
from backend.app.intelligence.documents.repository import DocumentRepository


MODEL_NAME = "test-embedding-model"
DIMENSIONS = 3


@pytest.fixture
def repository(tmp_path):
    repo = DocumentRepository(tmp_path / "documents.db")
    repo.initialize()
    return repo


def add_indexed_document(
    repository,
    *,
    scope=DocumentScope.TEMPORARY,
    conversation_id=None,
    expires_at=None,
    filename="notes.txt",
    content="Sample document content",
    vector=None,
):
    now = datetime.now(timezone.utc)

    if scope == DocumentScope.TEMPORARY:
        conversation_id = conversation_id or uuid4()
        expires_at = expires_at or now + timedelta(hours=24)
    else:
        conversation_id = None
        expires_at = None

    document_id = uuid4()
    chunk_id = uuid4()

    document = DocumentRecord(
        id=document_id,
        scope=scope,
        conversation_id=conversation_id,
        filename=filename,
        storage_key=f"{document_id}.txt",
        content_hash="a" * 64,
        mime_type="text/plain",
        size_bytes=len(content),
        created_at=now,
        expires_at=expires_at,
    )

    chunk = DocumentChunk(
        id=chunk_id,
        document_id=document_id,
        chunk_index=0,
        content=content,
        page_number=1,
    )

    embedding_vector = vector or [1.0, 0.0, 0.0]

    embedding = DocumentEmbedding(
        chunk_id=chunk_id,
        model_name=MODEL_NAME,
        dimensions=DIMENSIONS,
        vector=embedding_vector,
    )

    repository.create_document(document)
    repository.replace_index(
        document_id,
        [chunk],
        [embedding],
        page_count=1,
    )

    return document_id


def search(repository, *, scope, conversation_id=None, vector=None, **kwargs):
    return repository.search_chunks(
        vector or [1.0, 0.0, 0.0],
        model_name=MODEL_NAME,
        dimensions=DIMENSIONS,
        knowledge_scope=scope,
        conversation_id=conversation_id,
        **kwargs,
    )


def test_none_scope_returns_no_results(repository):
    add_indexed_document(repository, scope=DocumentScope.PERSISTENT)

    results = search(repository, scope="none")

    assert results == []


def test_conversation_scope_only_returns_own_documents(repository):
    conversation_a = uuid4()
    conversation_b = uuid4()

    own_document = add_indexed_document(
        repository,
        conversation_id=conversation_a,
        filename="own.txt",
    )

    add_indexed_document(
        repository,
        conversation_id=conversation_b,
        filename="other.txt",
    )

    results = search(
        repository,
        scope="conversation",
        conversation_id=conversation_a,
    )

    assert len(results) == 1
    assert results[0]["document_id"] == own_document
    assert results[0]["filename"] == "own.txt"


def test_conversation_scope_excludes_expired_documents(repository):
    conversation_id = uuid4()
    now = datetime.now(timezone.utc)

    add_indexed_document(
        repository,
        conversation_id=conversation_id,
        expires_at=now - timedelta(seconds=1),
        filename="expired.txt",
    )

    results = search(
        repository,
        scope="conversation",
        conversation_id=conversation_id,
        now=now,
    )

    assert results == []


def test_library_scope_only_returns_persistent_documents(repository):
    conversation_id = uuid4()

    persistent_document = add_indexed_document(
        repository,
        scope=DocumentScope.PERSISTENT,
        filename="library.txt",
    )

    add_indexed_document(
        repository,
        conversation_id=conversation_id,
        filename="temporary.txt",
    )

    results = search(repository, scope="library")

    assert len(results) == 1
    assert results[0]["document_id"] == persistent_document
    assert results[0]["filename"] == "library.txt"


def test_both_scope_returns_library_and_own_conversation(repository):
    conversation_a = uuid4()
    conversation_b = uuid4()

    library_document = add_indexed_document(
        repository,
        scope=DocumentScope.PERSISTENT,
        filename="library.txt",
    )

    own_document = add_indexed_document(
        repository,
        conversation_id=conversation_a,
        filename="own.txt",
    )

    add_indexed_document(
        repository,
        conversation_id=conversation_b,
        filename="other.txt",
    )

    results = search(
        repository,
        scope="both",
        conversation_id=conversation_a,
    )

    result_ids = {result["document_id"] for result in results}

    assert result_ids == {library_document, own_document}


def test_search_ranks_by_cosine_similarity(repository):
    add_indexed_document(
        repository,
        scope=DocumentScope.PERSISTENT,
        filename="less-relevant.txt",
        vector=[0.0, 1.0, 0.0],
    )

    best_document = add_indexed_document(
        repository,
        scope=DocumentScope.PERSISTENT,
        filename="most-relevant.txt",
        vector=[1.0, 0.0, 0.0],
    )

    results = search(repository, scope="library")

    assert results[0]["document_id"] == best_document
    assert results[0]["similarity"] == pytest.approx(1.0)


def test_search_returns_source_metadata(repository):
    document_id = add_indexed_document(
        repository,
        scope=DocumentScope.PERSISTENT,
        filename="research.pdf",
        content="Important research findings",
    )

    results = search(repository, scope="library")

    assert len(results) == 1
    assert results[0]["document_id"] == document_id
    assert results[0]["filename"] == "research.pdf"
    assert results[0]["chunk_index"] == 0
    assert results[0]["page_number"] == 1
    assert results[0]["content"] == "Important research findings"
    assert results[0]["similarity"] == pytest.approx(1.0)
    assert results[0]["chunk_id"] is not None


def test_search_ignores_incompatible_embedding_models(repository):
    document_id = add_indexed_document(
        repository,
        scope=DocumentScope.PERSISTENT,
    )

    with repository._connect() as connection:
        connection.execute(
            """
            UPDATE chunk_embeddings
            SET model_name = ?
            WHERE chunk_id IN (
                SELECT c.id
                FROM document_chunks AS c
                WHERE c.document_id = ?
            )
            """,
            ("different-model", str(document_id)),
        )

    results = search(repository, scope="library")

    assert results == []


def test_search_requires_conversation_id_for_conversation_scopes(repository):
    with pytest.raises(ValueError, match="conversation ID"):
        search(repository, scope="conversation")

    with pytest.raises(ValueError, match="conversation ID"):
        search(repository, scope="both")


def test_search_rejects_invalid_query_vectors(repository):
    with pytest.raises(ValueError, match="dimensions"):
        repository.search_chunks(
            [1.0, 0.0],
            model_name=MODEL_NAME,
            dimensions=DIMENSIONS,
            knowledge_scope="library",
        )

    with pytest.raises(ValueError, match="zero vector"):
        repository.search_chunks(
            [0.0, 0.0, 0.0],
            model_name=MODEL_NAME,
            dimensions=DIMENSIONS,
            knowledge_scope="library",
        )


def test_search_rejects_invalid_scope(repository):
    with pytest.raises(ValueError, match="knowledge scope"):
        search(repository, scope="everything")


def test_failed_index_replacement_preserves_existing_index(repository):
    import sqlite3

    document_id = add_indexed_document(
        repository,
        scope=DocumentScope.PERSISTENT,
        filename="original.txt",
        content="Original indexed knowledge must survive.",
    )

    original_document = repository.get_document(document_id)

    replacement_chunk = DocumentChunk(
        id=uuid4(),
        document_id=document_id,
        chunk_index=0,
        content="Replacement content that must not be committed.",
        page_number=9,
    )

    replacement_embedding = DocumentEmbedding(
        chunk_id=replacement_chunk.id,
        model_name=MODEL_NAME,
        dimensions=DIMENSIONS,
        vector=[0.0, 1.0, 0.0],
    )

    # Force a database failure after the old chunks have been deleted
    # and replacement chunks have been inserted.
    with repository._connect() as connection:
        connection.execute(
            """
            CREATE TRIGGER fail_replacement_embedding
            BEFORE INSERT ON chunk_embeddings
            BEGIN
                SELECT RAISE(
                    ABORT,
                    'simulated embedding insertion failure'
                );
            END;
            """
        )

    with pytest.raises(sqlite3.IntegrityError):
        repository.replace_index(
            document_id,
            [replacement_chunk],
            [replacement_embedding],
            page_count=9,
        )

    # The original document metadata must remain unchanged.
    restored_document = repository.get_document(document_id)

    assert restored_document is not None
    assert restored_document.status == DocumentStatus.INDEXED
    assert restored_document.page_count == original_document.page_count
    assert restored_document.chunk_count == original_document.chunk_count
    assert restored_document.indexed_at == original_document.indexed_at

    # The original chunk and embedding must still be searchable.
    results = search(repository, scope="library")

    assert len(results) == 1
    assert results[0]["document_id"] == document_id
    assert results[0]["content"] == (
        "Original indexed knowledge must survive."
    )
