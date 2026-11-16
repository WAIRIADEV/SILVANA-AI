
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.api.dependencies import (
    get_document_retrieval_service,
)
from backend.app.intelligence.documents.repository import (
    DocumentRepository,
)
from backend.app.intelligence.documents.retrieval import (
    DocumentRetrievalService,
)
from backend.app.intelligence.embeddings.fastembed import (
    FastEmbedProvider,
)


class FakeProviderManager:
    def __init__(self):
        self.closed = False

    async def aclose(self):
        self.closed = True


def create_fake_provider(monkeypatch):
    fake_manager = FakeProviderManager()

    monkeypatch.setattr(
        "backend.app.main.ProviderManager",
        lambda: fake_manager,
    )

    return fake_manager


def test_application_initializes_document_retrieval_service(
    monkeypatch,
):
    create_fake_provider(monkeypatch)

    with TestClient(app):
        assert isinstance(
            app.state.document_retrieval_service,
            DocumentRetrievalService,
        )


def test_retrieval_service_reuses_shared_dependencies(
    monkeypatch,
):
    create_fake_provider(monkeypatch)

    with TestClient(app):
        retrieval_service = app.state.document_retrieval_service

        assert isinstance(
            app.state.document_repository,
            DocumentRepository,
        )

        assert isinstance(
            app.state.embedding_provider,
            FastEmbedProvider,
        )

        assert (
            retrieval_service.repository
            is app.state.document_repository
        )

        assert (
            retrieval_service.embedding_provider
            is app.state.embedding_provider
        )

        assert (
            app.state.document_indexing_service.repository
            is app.state.document_repository
        )

        assert (
            app.state.document_indexing_service.embedding_provider
            is app.state.embedding_provider
        )


def test_retrieval_dependency_returns_shared_instance(
    monkeypatch,
):
    create_fake_provider(monkeypatch)

    with TestClient(app):
        request = type(
            "FakeRequest",
            (),
            {"app": app},
        )()

        assert (
            get_document_retrieval_service(request)
            is app.state.document_retrieval_service
        )
