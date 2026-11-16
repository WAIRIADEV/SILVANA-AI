
import hashlib
import logging
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path
from uuid import UUID, uuid4
from zipfile import BadZipFile, ZipFile

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
    status,
)

from backend.app.api.dependencies import (
    get_document_indexing_service,
    get_document_repository,
    get_document_storage,
)
from backend.app.api.schemas.documents import DocumentResponse
from backend.app.core.config import settings
from backend.app.intelligence.documents.indexing import (
    DocumentIndexingError,
    DocumentIndexingService,
)
from backend.app.intelligence.documents.models import (
    DocumentRecord,
    DocumentScope,
    DocumentStatus,
)
from backend.app.intelligence.documents.repository import DocumentRepository
from backend.app.intelligence.documents.storage import DocumentStorage


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents")

MAX_FILENAME_LENGTH = 255

SUPPORTED_MIME_TYPES = {
    ".pdf": "application/pdf",
    ".docx": (
        "application/vnd.openxmlformats-officedocument."
        "wordprocessingml.document"
    ),
    ".txt": "text/plain",
}


def _sanitize_filename(filename: str | None) -> str:
    if not filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A filename is required.",
        )

    normalized = filename.replace("\\", "/")
    safe_name = Path(normalized).name.strip()

    if (
        not safe_name
        or safe_name in {".", ".."}
        or "\x00" in safe_name
        or any(ord(character) < 32 for character in safe_name)
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid filename.",
        )

    if len(safe_name) > MAX_FILENAME_LENGTH:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename is too long.",
        )

    return safe_name


def _validate_document_content(
    filename: str,
    content: bytes,
) -> str:
    extension = Path(filename).suffix.lower()

    if extension not in SUPPORTED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only PDF, DOCX and TXT files are supported.",
        )

    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded file is empty.",
        )

    if extension == ".pdf":
        if not content.startswith(b"%PDF-"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="The file does not contain a valid PDF signature.",
            )

    elif extension == ".docx":
        try:
            with ZipFile(BytesIO(content)) as archive:
                names = set(archive.namelist())

                if (
                    "[Content_Types].xml" not in names
                    or "word/document.xml" not in names
                ):
                    raise ValueError("Invalid DOCX structure.")

                total_uncompressed_size = sum(
                    item.file_size for item in archive.infolist()
                )

                if total_uncompressed_size > 100 * 1024 * 1024:
                    raise ValueError("DOCX expands beyond the safe limit.")

        except (BadZipFile, ValueError, OSError):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="The file is not a valid DOCX document.",
            ) from None

    elif extension == ".txt":
        try:
            content.decode("utf-8-sig")
        except UnicodeDecodeError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="TXT documents must use UTF-8 encoding.",
            ) from None

    return SUPPORTED_MIME_TYPES[extension]


def _validate_scope(
    scope: DocumentScope,
    conversation_id: UUID | None,
) -> None:
    if (
        scope == DocumentScope.TEMPORARY
        and conversation_id is None
    ):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Temporary documents require a conversation ID.",
        )

    if (
        scope == DocumentScope.PERSISTENT
        and conversation_id is not None
    ):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Persistent documents cannot have a conversation ID.",
        )


def _get_accessible_document(
    document_id: UUID,
    scope: DocumentScope,
    conversation_id: UUID | None,
    repository: DocumentRepository,
) -> DocumentRecord:
    _validate_scope(scope, conversation_id)

    document = repository.get_document(document_id)

    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found.",
        )

    if document.scope != scope:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found.",
        )

    if (
        scope == DocumentScope.TEMPORARY
        and document.conversation_id != conversation_id
    ):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found.",
        )

    if (
        scope == DocumentScope.PERSISTENT
        and document.conversation_id is not None
    ):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found.",
        )

    return document


@router.post(
    "/upload",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_document(
    file: UploadFile = File(...),
    scope: DocumentScope = Form(...),
    conversation_id: UUID | None = Form(default=None),
    repository: DocumentRepository = Depends(get_document_repository),
    storage: DocumentStorage = Depends(get_document_storage),
    indexing_service: DocumentIndexingService = Depends(
        get_document_indexing_service
    ),
):
    """Store and index a document in its explicitly selected scope."""

    _validate_scope(scope, conversation_id)

    filename = _sanitize_filename(file.filename)

    try:
        content = await file.read(storage.max_file_size_bytes + 1)
    finally:
        await file.close()

    if len(content) > storage.max_file_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail="The uploaded file exceeds the maximum allowed size.",
        )

    mime_type = _validate_document_content(filename, content)

    document_id = uuid4()
    now = datetime.now(timezone.utc)

    expires_at = None

    if scope == DocumentScope.TEMPORARY:
        expires_at = now + timedelta(
            hours=settings.SILVANA_DOCUMENT_TEMPORARY_RETENTION_HOURS
        )

    try:
        storage_key = storage.save(
            content,
            document_id=document_id,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    document = DocumentRecord(
        id=document_id,
        scope=scope,
        conversation_id=conversation_id,
        filename=filename,
        storage_key=storage_key,
        content_hash=hashlib.sha256(content).hexdigest(),
        mime_type=mime_type,
        size_bytes=len(content),
        status=DocumentStatus.UPLOADED,
        created_at=now,
        last_activity_at=now if scope == DocumentScope.TEMPORARY else None,
        expires_at=expires_at,
    )

    try:
        repository.create_document(document)
    except Exception:
        try:
            storage.delete(storage_key)
        except Exception:
            logger.exception(
                "Failed to clean up document storage after database failure."
            )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The document could not be registered.",
        ) from None

    try:
        indexed_document = indexing_service.index_document(document_id)
        return DocumentResponse.model_validate(indexed_document)

    except DocumentIndexingError:
        logger.exception(
            "Document indexing failed for document %s.",
            document_id,
        )

        failed_document = repository.get_document(document_id)

        if failed_document is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="The uploaded document could not be retrieved.",
            ) from None

        # Keep the uploaded document so indexing can be retried.
        # Internal indexing errors remain private to the backend.
        return DocumentResponse.model_validate(failed_document)


@router.get(
    "",
    response_model=list[DocumentResponse],
)
def list_documents(
    scope: DocumentScope = Query(...),
    conversation_id: UUID | None = Query(default=None),
    repository: DocumentRepository = Depends(get_document_repository),
):
    """List documents visible within the explicitly selected scope."""

    _validate_scope(scope, conversation_id)

    documents = repository.list_documents(
        scope=scope,
        conversation_id=conversation_id,
    )

    return [
        DocumentResponse.model_validate(document)
        for document in documents
    ]


@router.get(
    "/{document_id}",
    response_model=DocumentResponse,
)
def get_document(
    document_id: UUID,
    scope: DocumentScope = Query(...),
    conversation_id: UUID | None = Query(default=None),
    repository: DocumentRepository = Depends(get_document_repository),
):
    """Retrieve metadata for a document visible to the requested scope."""

    document = _get_accessible_document(
        document_id,
        scope,
        conversation_id,
        repository,
    )

    return DocumentResponse.model_validate(document)


@router.post(
    "/{document_id}/reindex",
    response_model=DocumentResponse,
)
def reindex_document(
    document_id: UUID,
    scope: DocumentScope = Query(...),
    conversation_id: UUID | None = Query(default=None),
    repository: DocumentRepository = Depends(get_document_repository),
    indexing_service: DocumentIndexingService = Depends(
        get_document_indexing_service
    ),
):
    """Rebuild the index for an accessible document."""

    _get_accessible_document(
        document_id,
        scope,
        conversation_id,
        repository,
    )

    try:
        document = indexing_service.index_document(document_id)
        return DocumentResponse.model_validate(document)

    except DocumentIndexingError:
        logger.exception(
            "Re-indexing failed for document %s.",
            document_id,
        )

        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="The document could not be indexed.",
        ) from None


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_document(
    document_id: UUID,
    scope: DocumentScope = Query(...),
    conversation_id: UUID | None = Query(default=None),
    repository: DocumentRepository = Depends(get_document_repository),
    storage: DocumentStorage = Depends(get_document_storage),
):
    """Delete an accessible document and its stored bytes."""

    document = _get_accessible_document(
        document_id,
        scope,
        conversation_id,
        repository,
    )

    original_content = None

    try:
        if storage.exists(document.storage_key):
            original_content = storage.read(document.storage_key)
            storage.delete(document.storage_key)

        deleted = repository.delete_document(document_id)

        if not deleted:
            raise RuntimeError("Document record was not deleted.")

    except Exception:
        # Restore the physical file if an operation failed after
        # its contents were successfully read.
        if original_content is not None:
            try:
                if not storage.exists(document.storage_key):
                    storage.save(
                        original_content,
                        document_id=document_id,
                    )
            except Exception:
                logger.exception(
                    "CRITICAL: Failed to restore document storage "
                    "after deletion failure: %s",
                    document_id,
                )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The document could not be deleted.",
        ) from None
