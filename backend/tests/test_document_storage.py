
from uuid import UUID, uuid4

import pytest

from backend.app.intelligence.documents.storage import DocumentStorage


@pytest.fixture
def storage(tmp_path):
    """Provide isolated document storage for each test."""

    return DocumentStorage(
        storage_root=tmp_path / "documents",
        max_file_size_bytes=1024,
    )


def test_save_and_read_document(storage):
    content = b"SILVANA document storage test."

    storage_key = storage.save(content)

    assert storage.exists(storage_key)
    assert storage.read(storage_key) == content


def test_generated_storage_key_uses_uuid(storage):
    storage_key = storage.save(b"test content")

    identifier = storage_key.removesuffix(".document")

    assert storage_key.endswith(".document")
    assert len(identifier) == 32

    UUID(hex=identifier)


def test_save_uses_supplied_document_id(storage):
    document_id = uuid4()

    storage_key = storage.save(
        b"test content",
        document_id=document_id,
    )

    assert storage_key == f"{document_id.hex}.document"


def test_save_rejects_empty_content(storage):
    with pytest.raises(ValueError, match="cannot be empty"):
        storage.save(b"")


def test_save_rejects_non_bytes_content(storage):
    with pytest.raises(TypeError, match="must be bytes"):
        storage.save("not bytes")  # type: ignore[arg-type]


def test_save_rejects_oversized_content(storage):
    content = b"x" * 1025

    with pytest.raises(ValueError, match="maximum allowed size"):
        storage.save(content)


def test_save_accepts_content_at_size_limit(storage):
    content = b"x" * 1024

    storage_key = storage.save(content)

    assert storage.read(storage_key) == content


@pytest.mark.parametrize(
    "storage_key",
    [
        "../outside.txt",
        "..",
        ".",
        "",
        "folder/document.txt",
        "folder\\document.txt",
        "C:\\outside.txt",
    ],
)
def test_storage_rejects_unsafe_keys(storage, storage_key):
    with pytest.raises(ValueError, match="Invalid storage key"):
        storage.read(storage_key)


def test_duplicate_storage_key_does_not_overwrite(storage):
    document_id = uuid4()

    storage_key = storage.save(
        b"original content",
        document_id=document_id,
    )

    with pytest.raises(FileExistsError):
        storage.save(
            b"replacement content",
            document_id=document_id,
        )

    assert storage.read(storage_key) == b"original content"


def test_delete_removes_document(storage):
    storage_key = storage.save(b"test content")

    assert storage.delete(storage_key) is True
    assert storage.exists(storage_key) is False


def test_delete_returns_false_for_missing_document(storage):
    storage_key = f"{uuid4().hex}.document"

    assert storage.delete(storage_key) is False


def test_read_raises_for_missing_document(storage):
    storage_key = f"{uuid4().hex}.document"

    with pytest.raises(FileNotFoundError):
        storage.read(storage_key)


def test_storage_cleans_temporary_files_after_save(storage, tmp_path):
    storage_key = storage.save(b"test content")

    files = list((tmp_path / "documents").iterdir())

    assert len(files) == 1
    assert files[0].name == storage_key


def test_storage_rejects_invalid_maximum_file_size(tmp_path):
    with pytest.raises(ValueError, match="must be positive"):
        DocumentStorage(
            storage_root=tmp_path / "documents",
            max_file_size_bytes=0,
        )


def test_storage_creates_missing_storage_directory(tmp_path):
    storage_root = tmp_path / "nested" / "documents"

    DocumentStorage(storage_root)

    assert storage_root.is_dir()
