
from uuid import UUID

from backend.app.intelligence.documents.chunker import DocumentChunker
from backend.app.intelligence.documents.extractor import DocumentExtractor
from backend.app.intelligence.documents.models import (
    DocumentEmbedding,
    DocumentRecord,
    DocumentStatus,
)
from backend.app.intelligence.documents.repository import DocumentRepository
from backend.app.intelligence.documents.storage import DocumentStorage
from backend.app.intelligence.embeddings.base import EmbeddingProvider


class DocumentIndexingError(Exception):
    """Raised when a document cannot be indexed."""


class DocumentIndexingService:
    """Coordinates document extraction, chunking, embedding, and indexing."""

    def __init__(
        self,
        repository: DocumentRepository,
        storage: DocumentStorage,
        extractor: DocumentExtractor,
        chunker: DocumentChunker,
        embedding_provider: EmbeddingProvider,
    ) -> None:
        self.repository = repository
        self.storage = storage
        self.extractor = extractor
        self.chunker = chunker
        self.embedding_provider = embedding_provider

    def index_document(self, document_id: UUID) -> DocumentRecord:
        """Index a stored document and atomically replace its existing index."""

        document = self.repository.get_document(document_id)

        if document is None:
            raise ValueError("Document does not exist.")

        was_already_indexed = (
            document.status == DocumentStatus.INDEXED
        )

        # Preserve an existing valid index during re-indexing.
        # Only documents without a valid index enter PROCESSING.
        if not was_already_indexed:
            updated = self.repository.update_document_status(
                document_id,
                DocumentStatus.PROCESSING,
                error_message=None,
            )

            if not updated:
                raise ValueError(
                    "Document could not be moved into processing."
                )

        try:
            content = self.storage.read(document.storage_key)

            extraction = self.extractor.extract(
                document.filename,
                content,
            )

            chunks = self.chunker.chunk(
                extraction,
                document_id,
            )

            if not chunks:
                raise DocumentIndexingError(
                    "Document extraction produced no indexable chunks."
                )

            texts = [chunk.content for chunk in chunks]

            vectors = self.embedding_provider.embed_documents(texts)

            if len(vectors) != len(chunks):
                raise DocumentIndexingError(
                    "Embedding provider returned an unexpected "
                    "number of vectors."
                )

            embeddings = [
                DocumentEmbedding(
                    chunk_id=chunk.id,
                    model_name=self.embedding_provider.model_name,
                    dimensions=self.embedding_provider.dimensions,
                    vector=vector,
                )
                for chunk, vector in zip(chunks, vectors, strict=True)
            ]

            # The repository commits chunks, embeddings, and document
            # indexing metadata together in a single transaction.
            self.repository.replace_index(
                document_id,
                chunks,
                embeddings,
                page_count=extraction.page_count or 0,
            )

        except Exception as exc:
            # A failed first-time index should be marked FAILED.
            # A failed re-index must not overwrite the previous INDEXED
            # status or destroy its existing chunks and embeddings.
            if not was_already_indexed:
                try:
                    self.repository.update_document_status(
                        document_id,
                        DocumentStatus.FAILED,
                        error_message=str(exc)[:1000],
                    )
                except Exception:
                    # Preserve the original indexing exception.
                    pass

            if isinstance(exc, DocumentIndexingError):
                raise

            raise DocumentIndexingError(
                f"Document indexing failed: {exc}"
            ) from exc

        indexed_document = self.repository.get_document(document_id)

        if indexed_document is None:
            raise DocumentIndexingError(
                "Document was indexed but could not be retrieved."
            )

        if indexed_document.status != DocumentStatus.INDEXED:
            raise DocumentIndexingError(
                "Document indexing did not complete successfully."
            )

        return indexed_document

from uuid import UUID

from backend.app.intelligence.documents.chunker import DocumentChunker
from backend.app.intelligence.documents.extractor import DocumentExtractor
from backend.app.intelligence.documents.models import (
    DocumentEmbedding,
    DocumentRecord,
    DocumentStatus,
)
from backend.app.intelligence.documents.repository import DocumentRepository
from backend.app.intelligence.documents.storage import DocumentStorage
from backend.app.intelligence.embeddings.base import EmbeddingProvider


class DocumentIndexingError(Exception):
    """Raised when a document cannot be indexed."""


class DocumentIndexingService:
    """Coordinates document extraction, chunking, embedding, and indexing."""

    def __init__(
        self,
        repository: DocumentRepository,
        storage: DocumentStorage,
        extractor: DocumentExtractor,
        chunker: DocumentChunker,
        embedding_provider: EmbeddingProvider,
    ) -> None:
        self.repository = repository
        self.storage = storage
        self.extractor = extractor
        self.chunker = chunker
        self.embedding_provider = embedding_provider

    def index_document(self, document_id: UUID) -> DocumentRecord:
        """Index a stored document and atomically replace its existing index."""

        document = self.repository.get_document(document_id)

        if document is None:
            raise ValueError("Document does not exist.")

        was_already_indexed = (
            document.status == DocumentStatus.INDEXED
        )

        # Preserve an existing valid index during re-indexing.
        # Only documents without a valid index enter PROCESSING.
        if not was_already_indexed:
            updated = self.repository.update_document_status(
                document_id,
                DocumentStatus.PROCESSING,
                error_message=None,
            )

            if not updated:
                raise ValueError(
                    "Document could not be moved into processing."
                )

        try:
            content = self.storage.read(document.storage_key)

            extraction = self.extractor.extract(
                document.filename,
                content,
            )

            chunks = self.chunker.chunk(
                extraction,
                document_id,
            )

            if not chunks:
                raise DocumentIndexingError(
                    "Document extraction produced no indexable chunks."
                )

            texts = [chunk.content for chunk in chunks]

            vectors = self.embedding_provider.embed_documents(texts)

            if len(vectors) != len(chunks):
                raise DocumentIndexingError(
                    "Embedding provider returned an unexpected "
                    "number of vectors."
                )

            embeddings = [
                DocumentEmbedding(
                    chunk_id=chunk.id,
                    model_name=self.embedding_provider.model_name,
                    dimensions=self.embedding_provider.dimensions,
                    vector=vector,
                )
                for chunk, vector in zip(chunks, vectors, strict=True)
            ]

            # The repository commits chunks, embeddings, and document
            # indexing metadata together in a single transaction.
            self.repository.replace_index(
                document_id,
                chunks,
                embeddings,
                page_count=extraction.page_count or 0,
            )

        except Exception as exc:
            # A failed first-time index should be marked FAILED.
            # A failed re-index must not overwrite the previous INDEXED
            # status or destroy its existing chunks and embeddings.
            if not was_already_indexed:
                try:
                    self.repository.update_document_status(
                        document_id,
                        DocumentStatus.FAILED,
                        error_message=str(exc)[:1000],
                    )
                except Exception:
                    # Preserve the original indexing exception.
                    pass

            if isinstance(exc, DocumentIndexingError):
                raise

            raise DocumentIndexingError(
                f"Document indexing failed: {exc}"
            ) from exc

        indexed_document = self.repository.get_document(document_id)

        if indexed_document is None:
            raise DocumentIndexingError(
                "Document was indexed but could not be retrieved."
            )

        if indexed_document.status != DocumentStatus.INDEXED:
            raise DocumentIndexingError(
                "Document indexing did not complete successfully."
            )

        return indexed_document
