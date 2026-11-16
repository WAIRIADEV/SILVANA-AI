
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from backend.app.intelligence.documents.chunker import DocumentChunker
from backend.app.intelligence.documents.extractor import (
    DocumentExtractor,
)
from backend.app.intelligence.documents.indexing import (
    DocumentIndexingError,
    DocumentIndexingService,
)
from backend.app.intelligence.documents.models import (
    DocumentRecord,
    DocumentScope,
    DocumentStatus,
)
from backend.app.intelligence.documents.repository import (
    DocumentRepository,
)
from backend.app.intelligence.documents.storage import DocumentStorage
from backend.app.intelligence.embeddings.base import EmbeddingProvider


class FakeEmbeddingProvider(EmbeddingProvider):
    """Lightweight embedding provider for deterministic tests."""

    def __init__(
        self,
        *,
        vectors=None,
        model_name="fake-embedding-model",
    ):
        self.vectors = vectors
        self._model_name = model_name

    @property
    def provider_name(self):
        return "fake"

    @property
    def model_name(self):
        return self._model_name

    @property
    def dimensions(self):
        return 3

    def embed_documents(self, texts):
        if self.vectors is not None:
            return self.vectors

        return [
            [0.1, 0.2, 0.3]
            for _ in texts
        ]

    def embed_query(self, text):
        return [0.1, 0.2, 0.3]


class FailingExtractor:
    """Extractor that simulates document-processing failure."""

    def extract(self, filename, content):
        raise RuntimeError("Simulated extraction failure.")


@pytest.fixture
def document_environment(tmp_path):
    repository = DocumentRepository(
        tmp_path / "documents.db"
    )
    repository.initialize()

    storage = DocumentStorage(
        tmp_path / "storage"
    )

    document_id = uuid4()
    content = (
        b"SILVANA document indexing test. "
        b"This document contains useful information "
        b"for testing extraction, chunking, and embeddings."
    )

    storage_key = storage.save(
        content,
        document_id=document_id,
    )

    now = datetime.now(timezone.utc)

    document = DocumentRecord(
        id=document_id,
        scope=DocumentScope.PERSISTENT,
        conversation_id=None,
        filename="sample.txt",
        storage_key=storage_key,
        content_hash="a" * 64,
        mime_type="text/plain",
        size_bytes=len(content),
        created_at=now,
    )

    repository.create_document(document)

    return {
        "repository": repository,
        "storage": storage,
        "document": document,
        "extractor": DocumentExtractor(),
        "chunker": DocumentChunker(),
    }


def make_service(
    environment,
    *,
    extractor=None,
    embedding_provider=None,
):
    return DocumentIndexingService(
        repository=environment["repository"],
        storage=environment["storage"],
        extractor=extractor or environment["extractor"],
        chunker=environment["chunker"],
        embedding_provider=(
            embedding_provider or FakeEmbeddingProvider()
        ),
    )


def test_successfully_indexes_document(document_environment):
    service = make_service(document_environment)

    document = service.index_document(
        document_environment["document"].id
    )

    repository = document_environment["repository"]

    chunks = repository.list_chunks(document.id)
    embeddings = repository.list_embeddings(document.id)

    assert document.status == DocumentStatus.INDEXED
    assert document.chunk_count > 0
    assert document.indexed_at is not None

    assert len(chunks) == document.chunk_count
    assert len(embeddings) == len(chunks)

    assert all(
        embedding.model_name == "fake-embedding-model"
        for embedding in embeddings
    )


def test_failed_first_index_marks_document_failed(
    document_environment,
):
    service = make_service(
        document_environment,
        extractor=FailingExtractor(),
    )

    document_id = document_environment["document"].id

    with pytest.raises(
        DocumentIndexingError,
        match="Simulated extraction failure",
    ):
        service.index_document(document_id)

    document = document_environment["repository"].get_document(
        document_id
    )

    assert document.status == DocumentStatus.FAILED
    assert "Simulated extraction failure" in document.error_message


def test_failed_reindex_preserves_existing_index(
    document_environment,
):
    repository = document_environment["repository"]
    document_id = document_environment["document"].id

    # First create a valid index.
    successful_service = make_service(document_environment)
    original_document = successful_service.index_document(
        document_id
    )

    original_chunks = repository.list_chunks(document_id)
    original_embeddings = repository.list_embeddings(document_id)

    # Now simulate an extraction failure during re-indexing.
    failing_service = make_service(
        document_environment,
        extractor=FailingExtractor(),
    )

    with pytest.raises(DocumentIndexingError):
        failing_service.index_document(document_id)

    preserved_document = repository.get_document(document_id)
    preserved_chunks = repository.list_chunks(document_id)
    preserved_embeddings = repository.list_embeddings(document_id)

    assert preserved_document.status == DocumentStatus.INDEXED
    assert preserved_document.chunk_count == original_document.chunk_count

    assert preserved_chunks == original_chunks
    assert preserved_embeddings == original_embeddings


def test_embedding_count_mismatch_marks_first_index_failed(
    document_environment,
):
    provider = FakeEmbeddingProvider(vectors=[])

    service = make_service(
        document_environment,
        embedding_provider=provider,
    )

    document_id = document_environment["document"].id

    with pytest.raises(
        DocumentIndexingError,
        match="unexpected number of vectors",
    ):
        service.index_document(document_id)

    document = document_environment["repository"].get_document(
        document_id
    )

    assert document.status == DocumentStatus.FAILED
    assert document_environment["repository"].list_chunks(
        document_id
    ) == []


def test_reindex_uses_current_embedding_model(
    document_environment,
):
    repository = document_environment["repository"]
    document_id = document_environment["document"].id

    provider = FakeEmbeddingProvider(
        model_name="replacement-model"
    )

    service = make_service(
        document_environment,
        embedding_provider=provider,
    )

    document = service.index_document(document_id)

    embeddings = repository.list_embeddings(document_id)

    assert document.status == DocumentStatus.INDEXED
    assert embeddings
    assert all(
        embedding.model_name == "replacement-model"
        for embedding in embeddings
    )
