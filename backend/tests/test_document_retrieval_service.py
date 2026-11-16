
from uuid import uuid4

import pytest

from backend.app.intelligence.documents.models import DocumentScope
from backend.app.intelligence.documents.retrieval import (
    DocumentRetrievalResult,
    DocumentRetrievalService,
)


class FakeEmbeddingProvider:
    model_name = "test-embedding-model"
    dimensions = 4

    def __init__(self):
        self.queries = []

    def embed_query(self, text):
        self.queries.append(text)
        return [1.0, 0.0, 0.0, 0.0]


class FakeDocumentRepository:
    def __init__(self, results=None):
        self.results = results or []
        self.calls = []

    def search_chunks(self, query_vector, **kwargs):
        self.calls.append({
            "query_vector": query_vector,
            **kwargs,
        })
        return self.results


@pytest.fixture
def provider():
    return FakeEmbeddingProvider()


@pytest.fixture
def repository():
    document_id = uuid4()
    chunk_id = uuid4()

    return FakeDocumentRepository([
        {
            "document_id": document_id,
            "filename": "research.pdf",
            "scope": DocumentScope.PERSISTENT,
            "chunk_id": chunk_id,
            "chunk_index": 2,
            "content": "Machine learning research findings.",
            "page_number": 3,
            "similarity": 0.95,
        }
    ])


@pytest.fixture
def service(repository, provider):
    return DocumentRetrievalService(repository, provider)


def test_retrieve_returns_source_aware_results(service, repository, provider):
    results = service.retrieve(
        "Explain machine learning",
        knowledge_scope="library",
    )

    assert len(results) == 1

    result = results[0]

    assert isinstance(result, DocumentRetrievalResult)
    assert result.filename == "research.pdf"
    assert result.scope == DocumentScope.PERSISTENT
    assert result.content == "Machine learning research findings."
    assert result.page_number == 3
    assert result.similarity == 0.95

    assert provider.queries == ["Explain machine learning"]
    assert len(repository.calls) == 1


def test_retrieve_trims_query_before_embedding(service, provider):
    service.retrieve(
        "  Explain machine learning  ",
        knowledge_scope="library",
    )

    assert provider.queries == ["Explain machine learning"]


def test_retrieve_passes_embedding_configuration_to_repository(
    service,
    repository,
):
    service.retrieve(
        "Find relevant information",
        knowledge_scope="library",
    )

    call = repository.calls[0]

    assert call["query_vector"] == [1.0, 0.0, 0.0, 0.0]
    assert call["model_name"] == "test-embedding-model"
    assert call["dimensions"] == 4
    assert call["knowledge_scope"] == "library"
    assert call["limit"] == 5


def test_retrieve_passes_conversation_id_to_repository(
    service,
    repository,
):
    conversation_id = uuid4()

    service.retrieve(
        "Find conversation notes",
        knowledge_scope="conversation",
        conversation_id=conversation_id,
    )

    assert repository.calls[0]["conversation_id"] == conversation_id


def test_none_scope_skips_embedding_and_repository(service, repository, provider):
    results = service.retrieve(
        "This should not trigger retrieval",
        knowledge_scope="none",
    )

    assert results == []
    assert provider.queries == []
    assert repository.calls == []


def test_conversation_scope_requires_conversation_id(service):
    with pytest.raises(ValueError, match="conversation ID"):
        service.retrieve(
            "Find my uploaded document",
            knowledge_scope="conversation",
        )


def test_both_scope_requires_conversation_id(service):
    with pytest.raises(ValueError, match="conversation ID"):
        service.retrieve(
            "Search everything",
            knowledge_scope="both",
        )


def test_empty_query_is_rejected(service, provider, repository):
    with pytest.raises(ValueError, match="cannot be empty"):
        service.retrieve(
            "   ",
            knowledge_scope="library",
        )

    assert provider.queries == []
    assert repository.calls == []


def test_invalid_scope_is_rejected(service):
    with pytest.raises(ValueError, match="Unsupported knowledge scope"):
        service.retrieve(
            "Search documents",
            knowledge_scope="global",
        )


def test_invalid_limit_is_rejected(service):
    with pytest.raises(ValueError, match="greater than zero"):
        service.retrieve(
            "Search documents",
            knowledge_scope="library",
            limit=0,
        )
