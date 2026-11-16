
import sqlite3
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
from backend.app.intelligence.documents.repository import (
    DocumentRepository,
)


@pytest.fixture
def repository(tmp_path):
    repo = DocumentRepository(tmp_path / "documents.db")
    repo.initialize()
    return repo


@pytest.fixture
def document(repository):
    conversation_id = uuid4()

    record = DocumentRecord(
        id=uuid4(),
        scope=DocumentScope.TEMPORARY,
        conversation_id=conversation_id,
        filename="sample.txt",
        storage_key=f"{uuid4().hex}.document",
        content_hash="a" * 64,
        mime_type="text/plain",
        size_bytes=128,
        expires_at=datetime.now(timezone.utc) + timedelta(hours=24),
    )

    repository.create_document(record)
    return record


def make_chunk(document_id, index, content):
    return DocumentChunk(
        id=uuid4(),
        document_id=document_id,
        chunk_index=index,
        content=content,
        page_number=1,
    )


def make_embedding(
    chunk_id,
    vector=None,
    model_name="BAAI/bge-small-en-v1.5",
):
    vector = vector if vector is not None else [0.1, 0.2, 0.3, 0.4]

    return DocumentEmbedding(
        chunk_id=chunk_id,
        model_name=model_name,
        dimensions=len(vector),
        vector=vector,
    )


def test_replace_index_stores_chunks_and_embeddings(
    repository,
    document,
):
    chunks = [
        make_chunk(document.id, 0, "SILVANA is an AI assistant."),
        make_chunk(document.id, 1, "It can understand documents."),
    ]

    embeddings = [
        make_embedding(chunks[0].id, [0.1, 0.2, 0.3, 0.4]),
        make_embedding(chunks[1].id, [0.5, 0.6, 0.7, 0.8]),
    ]

    repository.replace_index(
        document.id,
        chunks,
        embeddings,
        page_count=1,
    )

    stored_document = repository.get_document(document.id)
    stored_chunks = repository.list_chunks(document.id)
    stored_embeddings = repository.list_embeddings(document.id)

    assert stored_document is not None
    assert stored_document.status == DocumentStatus.INDEXED
    assert stored_document.chunk_count == 2
    assert stored_document.page_count == 1
    assert stored_document.indexed_at is not None

    assert len(stored_chunks) == 2
    assert len(stored_embeddings) == 2


def test_embeddings_preserve_vectors_and_model_metadata(
    repository,
    document,
):
    chunk = make_chunk(
        document.id,
        0,
        "Example document content.",
    )

    expected_vector = [0.125, -0.25, 0.5, 0.75]

    embedding = make_embedding(
        chunk.id,
        expected_vector,
        model_name="test-embedding-model",
    )

    repository.replace_index(
        document.id,
        [chunk],
        [embedding],
        page_count=1,
    )

    stored = repository.list_embeddings(document.id)

    assert len(stored) == 1
    assert stored[0].chunk_id == chunk.id
    assert stored[0].model_name == "test-embedding-model"
    assert stored[0].dimensions == 4
    assert stored[0].vector == pytest.approx(expected_vector)


def test_replace_index_replaces_previous_chunks_and_embeddings(
    repository,
    document,
):
    old_chunk = make_chunk(
        document.id,
        0,
        "Old document content.",
    )

    old_embedding = make_embedding(
        old_chunk.id,
        [0.1, 0.2, 0.3, 0.4],
    )

    repository.replace_index(
        document.id,
        [old_chunk],
        [old_embedding],
        page_count=1,
    )

    new_chunks = [
        make_chunk(document.id, 0, "Updated first section."),
        make_chunk(document.id, 1, "Updated second section."),
    ]

    new_embeddings = [
        make_embedding(new_chunks[0].id, [0.5, 0.6, 0.7, 0.8]),
        make_embedding(new_chunks[1].id, [0.9, 0.8, 0.7, 0.6]),
    ]

    repository.replace_index(
        document.id,
        new_chunks,
        new_embeddings,
        page_count=2,
    )

    stored_chunks = repository.list_chunks(document.id)
    stored_embeddings = repository.list_embeddings(document.id)

    assert len(stored_chunks) == 2
    assert len(stored_embeddings) == 2

    assert old_chunk.id not in {
        chunk.id for chunk in stored_chunks
    }

    assert old_chunk.id not in {
        embedding.chunk_id for embedding in stored_embeddings
    }

    assert [chunk.content for chunk in stored_chunks] == [
        "Updated first section.",
        "Updated second section.",
    ]


def test_failed_reindex_preserves_previous_index(
    repository,
    document,
):
    original_chunk = make_chunk(
        document.id,
        0,
        "Original indexed content.",
    )

    original_embedding = make_embedding(
        original_chunk.id,
        [0.1, 0.2, 0.3, 0.4],
    )

    repository.replace_index(
        document.id,
        [original_chunk],
        [original_embedding],
        page_count=1,
    )

    replacement_chunks = [
        make_chunk(document.id, 0, "Replacement content."),
        make_chunk(document.id, 0, "Duplicate chunk index."),
    ]

    replacement_embeddings = [
        make_embedding(
            replacement_chunks[0].id,
            [0.5, 0.6, 0.7, 0.8],
        ),
        make_embedding(
            replacement_chunks[1].id,
            [0.9, 0.8, 0.7, 0.6],
        ),
    ]

    with pytest.raises(sqlite3.IntegrityError):
        repository.replace_index(
            document.id,
            replacement_chunks,
            replacement_embeddings,
            page_count=2,
        )

    stored_chunks = repository.list_chunks(document.id)
    stored_embeddings = repository.list_embeddings(document.id)
    stored_document = repository.get_document(document.id)

    assert stored_document is not None
    assert stored_document.status == DocumentStatus.INDEXED

    assert len(stored_chunks) == 1
    assert stored_chunks[0].id == original_chunk.id
    assert stored_chunks[0].content == "Original indexed content."

    assert len(stored_embeddings) == 1
    assert stored_embeddings[0].chunk_id == original_chunk.id
    assert stored_embeddings[0].vector == pytest.approx(
        original_embedding.vector
    )


@pytest.mark.parametrize(
    "invalid_vector",
    [
        [],
        [0.1, float("nan"), 0.3],
        [0.1, float("inf"), 0.3],
        [0.1, float("-inf"), 0.3],
    ],
)
def test_replace_index_rejects_invalid_vectors(
    repository,
    document,
    invalid_vector,
):
    chunk = make_chunk(
        document.id,
        0,
        "Example content.",
    )

    embedding = DocumentEmbedding(
        chunk_id=chunk.id,
        model_name="test-model",
        dimensions=len(invalid_vector),
        vector=invalid_vector,
    )

    with pytest.raises((ValueError, TypeError)):
        repository.replace_index(
            document.id,
            [chunk],
            [embedding],
            page_count=1,
        )

    assert repository.list_chunks(document.id) == []
    assert repository.list_embeddings(document.id) == []


def test_replace_index_rejects_dimension_mismatch(
    repository,
    document,
):
    chunk = make_chunk(
        document.id,
        0,
        "Example content.",
    )

    embedding = DocumentEmbedding(
        chunk_id=chunk.id,
        model_name="test-model",
        dimensions=384,
        vector=[0.1, 0.2, 0.3, 0.4],
    )

    with pytest.raises(ValueError):
        repository.replace_index(
            document.id,
            [chunk],
            [embedding],
            page_count=1,
        )

    assert repository.list_chunks(document.id) == []
    assert repository.list_embeddings(document.id) == []


def test_replace_index_rejects_embedding_without_matching_chunk(
    repository,
    document,
):
    chunk = make_chunk(
        document.id,
        0,
        "Example content.",
    )

    embedding = make_embedding(uuid4())

    with pytest.raises(ValueError):
        repository.replace_index(
            document.id,
            [chunk],
            [embedding],
            page_count=1,
        )

    assert repository.list_chunks(document.id) == []
    assert repository.list_embeddings(document.id) == []


def test_replace_index_rejects_chunks_from_another_document(
    repository,
    document,
):
    foreign_chunk = make_chunk(
        uuid4(),
        0,
        "Foreign document content.",
    )

    embedding = make_embedding(foreign_chunk.id)

    with pytest.raises(ValueError):
        repository.replace_index(
            document.id,
            [foreign_chunk],
            [embedding],
            page_count=1,
        )

    assert repository.list_chunks(document.id) == []
    assert repository.list_embeddings(document.id) == []


def test_delete_document_cascades_to_embeddings(
    repository,
    document,
):
    chunk = make_chunk(
        document.id,
        0,
        "Content to delete.",
    )

    embedding = make_embedding(chunk.id)

    repository.replace_index(
        document.id,
        [chunk],
        [embedding],
        page_count=1,
    )

    assert len(repository.list_embeddings(document.id)) == 1

    assert repository.delete_document(document.id) is True

    assert repository.get_document(document.id) is None
    assert repository.list_chunks(document.id) == []
    assert repository.list_embeddings(document.id) == []


def test_list_embeddings_returns_empty_for_unindexed_document(
    repository,
    document,
):
    assert repository.list_embeddings(document.id) == []


def test_replace_index_rejects_missing_document(repository):
    chunk = make_chunk(
        uuid4(),
        0,
        "Orphan content.",
    )

    embedding = make_embedding(chunk.id)

    with pytest.raises(ValueError):
        repository.replace_index(
            chunk.document_id,
            [chunk],
            [embedding],
            page_count=1,
        )
