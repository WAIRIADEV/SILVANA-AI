
import io
from types import SimpleNamespace
from uuid import uuid4
from zipfile import ZIP_DEFLATED, ZipFile

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.api.dependencies import (
    get_document_indexing_service,
    get_document_repository,
    get_document_storage,
)
from backend.app.intelligence.documents.indexing import (
    DocumentIndexingError,
)
from backend.app.intelligence.documents.models import DocumentStatus
from backend.app.intelligence.documents.repository import DocumentRepository
from backend.app.intelligence.documents.storage import DocumentStorage


class FakeIndexingService:
    """Controlled indexing double backed by the real test repository."""

    def __init__(self, repository):
        self.repository = repository
        self.calls = []
        self.fail = False

    def index_document(self, document_id):
        self.calls.append(document_id)

        if self.fail:
            self.repository.update_document_status(
                document_id,
                status=DocumentStatus.FAILED,
                error_message="PRIVATE INTERNAL INDEXING ERROR",
                page_count=0,
                chunk_count=0,
            )
            raise DocumentIndexingError(
                "PRIVATE INTERNAL INDEXING ERROR"
            )

        self.repository.update_document_status(
            document_id,
            status=DocumentStatus.INDEXED,
            error_message=None,
            page_count=1,
            chunk_count=1,
        )

        document = self.repository.get_document(document_id)

        if document is None:
            raise RuntimeError("Test document disappeared.")

        return document


@pytest.fixture
def document_api(monkeypatch, tmp_path):
    """Provide isolated real storage and SQLite for each API test."""

    repository = DocumentRepository(
        tmp_path / "documents.sqlite3"
    )
    repository.initialize()

    storage = DocumentStorage(
        tmp_path / "document-storage",
        max_file_size_bytes=1024,
    )

    indexing_service = FakeIndexingService(repository)

    monkeypatch.setitem(
        app.dependency_overrides,
        get_document_repository,
        lambda: repository,
    )
    monkeypatch.setitem(
        app.dependency_overrides,
        get_document_storage,
        lambda: storage,
    )
    monkeypatch.setitem(
        app.dependency_overrides,
        get_document_indexing_service,
        lambda: indexing_service,
    )

    with TestClient(app) as client:
        yield SimpleNamespace(
            client=client,
            repository=repository,
            storage=storage,
            indexing=indexing_service,
        )


def make_docx():
    """Build a minimal structurally valid DOCX archive."""

    output = io.BytesIO()

    with ZipFile(output, "w", ZIP_DEFLATED) as archive:
        archive.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/'
            'package/2006/content-types"></Types>',
        )
        archive.writestr(
            "word/document.xml",
            '<?xml version="1.0"?>'
            '<w:document xmlns:w="http://schemas.openxmlformats.org/'
            'wordprocessingml/2006/main">'
            '<w:body><w:p><w:r><w:t>Test document</w:t>'
            '</w:r></w:p></w:body></w:document>',
        )

    return output.getvalue()


def upload_temporary(
    client,
    *,
    conversation_id=None,
    filename="notes.txt",
    content=b"SILVANA document API test content.",
):
    conversation_id = conversation_id or uuid4()

    response = client.post(
        "/api/v1/documents/upload",
        data={
            "scope": "temporary",
            "conversation_id": str(conversation_id),
        },
        files={
            "file": (
                filename,
                content,
                "application/octet-stream",
            )
        },
    )

    return response, conversation_id


def test_uploads_supported_document_formats(document_api):
    formats = [
        (
            "notes.txt",
            b"Valid UTF-8 document content.",
            "text/plain",
        ),
        (
            "report.pdf",
            b"%PDF-1.4\nTest PDF content.",
            "application/pdf",
        ),
        (
            "report.docx",
            make_docx(),
            "application/vnd.openxmlformats-officedocument."
            "wordprocessingml.document",
        ),
    ]

    for filename, content, expected_mime in formats:
        response, conversation_id = upload_temporary(
            document_api.client,
            filename=filename,
            content=content,
        )

        assert response.status_code == 201

        data = response.json()

        assert data["filename"] == filename
        assert data["mime_type"] == expected_mime
        assert data["scope"] == "temporary"
        assert data["conversation_id"] == str(conversation_id)
        assert data["status"] == "indexed"

        assert "storage_key" not in data
        assert "error_message" not in data


def test_upload_persists_document_and_file(document_api):
    response, conversation_id = upload_temporary(
        document_api.client
    )

    assert response.status_code == 201

    data = response.json()
    document_id = data["id"]

    stored_document = document_api.repository.get_document(
        document_id
    )

    assert stored_document is not None
    assert stored_document.conversation_id == conversation_id
    assert document_api.storage.exists(
        stored_document.storage_key
    )

    assert document_api.storage.read(
        stored_document.storage_key
    ) == b"SILVANA document API test content."


def test_upload_requires_conversation_id_for_temporary_scope(
    document_api,
):
    response = document_api.client.post(
        "/api/v1/documents/upload",
        data={"scope": "temporary"},
        files={
            "file": (
                "notes.txt",
                b"Some content.",
                "text/plain",
            )
        },
    )

    assert response.status_code == 422


def test_persistent_upload_rejects_conversation_id(document_api):
    response = document_api.client.post(
        "/api/v1/documents/upload",
        data={
            "scope": "persistent",
            "conversation_id": str(uuid4()),
        },
        files={
            "file": (
                "notes.txt",
                b"Some content.",
                "text/plain",
            )
        },
    )

    assert response.status_code == 422


@pytest.mark.parametrize(
    ("filename", "content", "expected_status"),
    [
        ("empty.txt", b"", 400),
        ("notes.exe", b"not supported", 415),
        ("broken.pdf", b"not a PDF", 400),
        ("broken.docx", b"not a DOCX archive", 400),
        ("invalid.txt", b"\xff\xfe\xfa", 400),
    ],
)
def test_upload_rejects_invalid_documents(
    document_api,
    filename,
    content,
    expected_status,
):
    response, _ = upload_temporary(
        document_api.client,
        filename=filename,
        content=content,
    )

    assert response.status_code == expected_status


def test_upload_rejects_oversized_document(document_api):
    response, _ = upload_temporary(
        document_api.client,
        content=b"x" * 1025,
    )

    assert response.status_code == 413


def test_upload_sanitizes_client_filename(document_api):
    response, _ = upload_temporary(
        document_api.client,
        filename=r"..\private\notes.txt",
    )

    assert response.status_code == 201
    assert response.json()["filename"] == "notes.txt"


def test_temporary_documents_are_isolated_by_conversation(
    document_api,
):
    first_conversation = uuid4()
    second_conversation = uuid4()

    response, _ = upload_temporary(
        document_api.client,
        conversation_id=first_conversation,
    )

    assert response.status_code == 201
    document_id = response.json()["id"]

    first_list = document_api.client.get(
        "/api/v1/documents",
        params={
            "scope": "temporary",
            "conversation_id": str(first_conversation),
        },
    )

    second_list = document_api.client.get(
        "/api/v1/documents",
        params={
            "scope": "temporary",
            "conversation_id": str(second_conversation),
        },
    )

    assert first_list.status_code == 200
    assert second_list.status_code == 200

    assert len(first_list.json()) == 1
    assert second_list.json() == []

    inaccessible = document_api.client.get(
        f"/api/v1/documents/{document_id}",
        params={
            "scope": "temporary",
            "conversation_id": str(second_conversation),
        },
    )

    assert inaccessible.status_code == 404


def test_temporary_document_requires_conversation_for_listing(
    document_api,
):
    response = document_api.client.get(
        "/api/v1/documents",
        params={"scope": "temporary"},
    )

    assert response.status_code == 422


def test_persistent_documents_can_be_listed_without_conversation(
    document_api,
):
    response = document_api.client.post(
        "/api/v1/documents/upload",
        data={"scope": "persistent"},
        files={
            "file": (
                "library.txt",
                b"Persistent knowledge library content.",
                "text/plain",
            )
        },
    )

    assert response.status_code == 201
    document_id = response.json()["id"]

    listing = document_api.client.get(
        "/api/v1/documents",
        params={"scope": "persistent"},
    )

    assert listing.status_code == 200
    assert len(listing.json()) == 1
    assert listing.json()[0]["id"] == document_id
    assert listing.json()[0]["conversation_id"] is None


def test_document_metadata_is_not_accessible_under_wrong_scope(
    document_api,
):
    response, conversation_id = upload_temporary(
        document_api.client
    )

    document_id = response.json()["id"]

    inaccessible = document_api.client.get(
        f"/api/v1/documents/{document_id}",
        params={"scope": "persistent"},
    )

    assert inaccessible.status_code == 404


def test_successful_upload_invokes_indexing(document_api):
    response, _ = upload_temporary(document_api.client)

    assert response.status_code == 201
    assert len(document_api.indexing.calls) == 1
    assert str(document_api.indexing.calls[0]) == response.json()["id"]


def test_failed_upload_indexing_preserves_document_safely(
    document_api,
):
    document_api.indexing.fail = True

    response, conversation_id = upload_temporary(
        document_api.client
    )

    assert response.status_code == 201

    data = response.json()

    assert data["status"] == "failed"
    assert data["conversation_id"] == str(conversation_id)
    assert "error_message" not in data
    assert "PRIVATE INTERNAL INDEXING ERROR" not in response.text
    assert "storage_key" not in data

    stored_document = document_api.repository.get_document(
        data["id"]
    )

    assert stored_document is not None
    assert stored_document.status == DocumentStatus.FAILED
    assert document_api.storage.exists(
        stored_document.storage_key
    )


def test_get_document_returns_safe_metadata(document_api):
    response, conversation_id = upload_temporary(
        document_api.client
    )

    document_id = response.json()["id"]

    metadata = document_api.client.get(
        f"/api/v1/documents/{document_id}",
        params={
            "scope": "temporary",
            "conversation_id": str(conversation_id),
        },
    )

    assert metadata.status_code == 200

    data = metadata.json()

    assert data["id"] == document_id
    assert "storage_key" not in data
    assert "error_message" not in data


def test_reindex_document(document_api):
    response, conversation_id = upload_temporary(
        document_api.client
    )

    document_id = response.json()["id"]
    initial_call_count = len(document_api.indexing.calls)

    reindex_response = document_api.client.post(
        f"/api/v1/documents/{document_id}/reindex",
        params={
            "scope": "temporary",
            "conversation_id": str(conversation_id),
        },
    )

    assert reindex_response.status_code == 200
    assert reindex_response.json()["status"] == "indexed"
    assert len(document_api.indexing.calls) == initial_call_count + 1


def test_reindex_rejects_wrong_conversation(document_api):
    response, _ = upload_temporary(document_api.client)

    document_id = response.json()["id"]

    reindex_response = document_api.client.post(
        f"/api/v1/documents/{document_id}/reindex",
        params={
            "scope": "temporary",
            "conversation_id": str(uuid4()),
        },
    )

    assert reindex_response.status_code == 404


def test_reindex_failure_returns_safe_error(document_api):
    response, conversation_id = upload_temporary(
        document_api.client
    )

    document_id = response.json()["id"]
    document_api.indexing.fail = True

    reindex_response = document_api.client.post(
        f"/api/v1/documents/{document_id}/reindex",
        params={
            "scope": "temporary",
            "conversation_id": str(conversation_id),
        },
    )

    assert reindex_response.status_code == 422
    assert reindex_response.json()["detail"] == (
        "The document could not be indexed."
    )
    assert "PRIVATE INTERNAL INDEXING ERROR" not in (
        reindex_response.text
    )


def test_delete_document_removes_database_record_and_file(
    document_api,
):
    response, conversation_id = upload_temporary(
        document_api.client
    )

    document_id = response.json()["id"]
    stored_document = document_api.repository.get_document(
        document_id
    )

    assert stored_document is not None
    assert document_api.storage.exists(
        stored_document.storage_key
    )

    delete_response = document_api.client.delete(
        f"/api/v1/documents/{document_id}",
        params={
            "scope": "temporary",
            "conversation_id": str(conversation_id),
        },
    )

    assert delete_response.status_code == 204
    assert document_api.repository.get_document(document_id) is None
    assert not document_api.storage.exists(
        stored_document.storage_key
    )


def test_delete_rejects_wrong_conversation(document_api):
    response, _ = upload_temporary(document_api.client)

    document_id = response.json()["id"]

    delete_response = document_api.client.delete(
        f"/api/v1/documents/{document_id}",
        params={
            "scope": "temporary",
            "conversation_id": str(uuid4()),
        },
    )

    assert delete_response.status_code == 404
    assert document_api.repository.get_document(document_id) is not None

def test_delete_storage_failure_preserves_database_record(
    document_api,
    monkeypatch,
):
    response, conversation_id = upload_temporary(
        document_api.client
    )

    document_id = response.json()["id"]
    stored_document = document_api.repository.get_document(
        document_id
    )

    def fail_delete(storage_key):
        raise OSError("Simulated storage failure")

    monkeypatch.setattr(
        document_api.storage,
        "delete",
        fail_delete,
    )

    delete_response = document_api.client.delete(
        f"/api/v1/documents/{document_id}",
        params={
            "scope": "temporary",
            "conversation_id": str(conversation_id),
        },
    )

    assert delete_response.status_code == 500
    assert document_api.repository.get_document(document_id) is not None
    assert document_api.storage.exists(stored_document.storage_key)


def test_delete_database_failure_restores_storage_file(
    document_api,
    monkeypatch,
):
    response, conversation_id = upload_temporary(
        document_api.client
    )

    document_id = response.json()["id"]
    stored_document = document_api.repository.get_document(
        document_id
    )

    original_content = document_api.storage.read(
        stored_document.storage_key
    )

    def fail_database_delete(document_id):
        raise RuntimeError("Simulated database failure")

    monkeypatch.setattr(
        document_api.repository,
        "delete_document",
        fail_database_delete,
    )

    delete_response = document_api.client.delete(
        f"/api/v1/documents/{document_id}",
        params={
            "scope": "temporary",
            "conversation_id": str(conversation_id),
        },
    )

    assert delete_response.status_code == 500
    assert document_api.repository.get_document(document_id) is not None
    assert document_api.storage.exists(stored_document.storage_key)
    assert document_api.storage.read(stored_document.storage_key) == original_content
