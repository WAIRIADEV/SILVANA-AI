
import math
import sqlite3
import struct
from collections import defaultdict
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID

from backend.app.intelligence.documents.models import (
    DocumentChunk,
    DocumentEmbedding,
    DocumentRecord,
    DocumentScope,
    DocumentStatus,
)


class DocumentRepository:
    """SQLite repository for documents, chunks, and vector embeddings."""

    def __init__(self, database_path: Path | str) -> None:
        self.database_path = Path(database_path)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)

        connection = sqlite3.connect(
            self.database_path,
            timeout=30,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")

        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        """Create repository tables and indexes without destroying data."""
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")

            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    scope TEXT NOT NULL
                        CHECK (scope IN ('temporary', 'persistent')),
                    conversation_id TEXT,
                    filename TEXT NOT NULL,
                    storage_key TEXT NOT NULL UNIQUE,
                    content_hash TEXT NOT NULL,
                    mime_type TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL CHECK (size_bytes >= 0),
                    status TEXT NOT NULL DEFAULT 'uploaded',
                    page_count INTEGER NOT NULL DEFAULT 0,
                    chunk_count INTEGER NOT NULL DEFAULT 0,
                    error_message TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    indexed_at TEXT,
                    last_accessed_at TEXT,
                    expires_at TEXT,
                    CHECK (
                        (
                            scope = 'temporary'
                            AND conversation_id IS NOT NULL
                            AND expires_at IS NOT NULL
                        )
                        OR
                        (
                            scope = 'persistent'
                            AND conversation_id IS NULL
                            AND expires_at IS NULL
                        )
                    )
                )
                """
            )

            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    page_number INTEGER,
                    start_char INTEGER,
                    end_char INTEGER,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (document_id)
                        REFERENCES documents(id)
                        ON DELETE CASCADE,
                    UNIQUE (document_id, chunk_index)
                )
                """
            )

            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS chunk_embeddings (
                    chunk_id TEXT PRIMARY KEY,
                    model_name TEXT NOT NULL,
                    dimensions INTEGER NOT NULL CHECK (dimensions > 0),
                    vector BLOB NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (chunk_id)
                        REFERENCES document_chunks(id)
                        ON DELETE CASCADE
                )
                """
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_documents_conversation
                ON documents(conversation_id)
                """
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_documents_expiry
                ON documents(expires_at)
                """
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_documents_scope_status
                ON documents(scope, status)
                """
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_chunks_document
                ON document_chunks(document_id, chunk_index)
                """
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_embeddings_model
                ON chunk_embeddings(model_name)
                """
            )

    @staticmethod
    def _utc_now() -> datetime:
        return datetime.now(timezone.utc)

    @staticmethod
    def _validate_datetime(value: datetime | None) -> None:
        if value is None:
            return

        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Datetime values must be timezone-aware.")

    @classmethod
    def _serialize_datetime(cls, value: datetime | None) -> str | None:
        if value is None:
            return None

        cls._validate_datetime(value)
        return value.astimezone(timezone.utc).isoformat()

    @staticmethod
    def _deserialize_datetime(value: str | None) -> datetime | None:
        if value is None:
            return None

        parsed = datetime.fromisoformat(value)

        if parsed.tzinfo is None or parsed.utcoffset() is None:
            parsed = parsed.replace(tzinfo=timezone.utc)

        return parsed.astimezone(timezone.utc)

    @classmethod
    def _document_from_row(
        cls,
        row: sqlite3.Row,
    ) -> DocumentRecord:
        return DocumentRecord(
            id=UUID(row["id"]),
            scope=DocumentScope(row["scope"]),
            conversation_id=(
                UUID(row["conversation_id"])
                if row["conversation_id"]
                else None
            ),
            filename=row["filename"],
            storage_key=row["storage_key"],
            content_hash=row["content_hash"],
            mime_type=row["mime_type"],
            size_bytes=row["size_bytes"],
            status=DocumentStatus(row["status"]),
            page_count=row["page_count"],
            chunk_count=row["chunk_count"],
            error_message=row["error_message"],
            created_at=cls._deserialize_datetime(row["created_at"]),
            last_activity_at=cls._deserialize_datetime(
                row["last_accessed_at"]
            ),
            expires_at=cls._deserialize_datetime(row["expires_at"]),
            indexed_at=cls._deserialize_datetime(row["indexed_at"]),
        )

    @classmethod
    def _chunk_from_row(cls, row: sqlite3.Row) -> DocumentChunk:
        return DocumentChunk(
            id=UUID(row["id"]),
            document_id=UUID(row["document_id"]),
            chunk_index=row["chunk_index"],
            content=row["content"],
            page_number=row["page_number"],
            created_at=cls._deserialize_datetime(row["created_at"]),
        )

    @classmethod
    def _embedding_from_row(
        cls,
        row: sqlite3.Row,
    ) -> DocumentEmbedding:
        dimensions = row["dimensions"]
        vector_blob = row["vector"]

        expected_bytes = dimensions * 4

        if len(vector_blob) != expected_bytes:
            raise ValueError(
                "Stored embedding vector has an invalid byte length."
            )

        vector = list(struct.unpack(f"<{dimensions}f", vector_blob))

        if not all(math.isfinite(value) for value in vector):
            raise ValueError(
                "Stored embedding contains non-finite values."
            )

        return DocumentEmbedding(
            chunk_id=UUID(row["chunk_id"]),
            model_name=row["model_name"],
            dimensions=dimensions,
            vector=vector,
            created_at=cls._deserialize_datetime(row["created_at"]),
        )

    @classmethod
    def _validate_and_encode_embedding(
        cls,
        embedding: DocumentEmbedding,
    ) -> bytes:
        if not embedding.model_name or not embedding.model_name.strip():
            raise ValueError("Embedding model name cannot be empty.")

        if embedding.dimensions <= 0:
            raise ValueError("Embedding dimensions must be positive.")

        if len(embedding.vector) != embedding.dimensions:
            raise ValueError(
                "Embedding dimensions do not match vector length."
            )

        cls._validate_datetime(embedding.created_at)

        try:
            values = [float(value) for value in embedding.vector]
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Embedding contains invalid values.") from exc

        if not all(math.isfinite(value) for value in values):
            raise ValueError(
                "Embedding vectors must contain only finite values."
            )

        try:
            encoded = struct.pack(
                f"<{embedding.dimensions}f",
                *values,
            )
        except (struct.error, OverflowError) as exc:
            raise ValueError(
                "Embedding values cannot be represented as float32."
            ) from exc

        decoded = struct.unpack(
            f"<{embedding.dimensions}f",
            encoded,
        )

        if not all(math.isfinite(value) for value in decoded):
            raise ValueError(
                "Embedding values overflowed during float32 conversion."
            )

        return encoded

    def create_document(self, document: DocumentRecord) -> None:
        now_datetime = self._utc_now()

        self._validate_datetime(document.created_at)
        self._validate_datetime(document.last_activity_at)
        self._validate_datetime(document.expires_at)
        self._validate_datetime(document.indexed_at)

        created_datetime = document.created_at or now_datetime

        last_activity_datetime = document.last_activity_at
        if (
            last_activity_datetime is None
            and document.scope == DocumentScope.TEMPORARY
        ):
            last_activity_datetime = created_datetime

        now = self._serialize_datetime(now_datetime)
        created_at = self._serialize_datetime(created_datetime)

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO documents (
                    id,
                    scope,
                    conversation_id,
                    filename,
                    storage_key,
                    content_hash,
                    mime_type,
                    size_bytes,
                    status,
                    page_count,
                    chunk_count,
                    error_message,
                    created_at,
                    updated_at,
                    indexed_at,
                    last_accessed_at,
                    expires_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(document.id),
                    document.scope.value,
                    (
                        str(document.conversation_id)
                        if document.conversation_id
                        else None
                    ),
                    document.filename,
                    document.storage_key,
                    document.content_hash,
                    document.mime_type,
                    document.size_bytes,
                    document.status.value,
                    document.page_count or 0,
                    document.chunk_count,
                    document.error_message,
                    created_at,
                    created_at or now,
                    self._serialize_datetime(document.indexed_at),
                    self._serialize_datetime(last_activity_datetime),
                    self._serialize_datetime(document.expires_at),
                ),
            )

    def get_document(self, document_id: UUID) -> DocumentRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM documents WHERE id = ?",
                (str(document_id),),
            ).fetchone()

        return self._document_from_row(row) if row else None

    def list_documents(
        self,
        *,
        scope: DocumentScope | None = None,
        conversation_id: UUID | None = None,
        status: DocumentStatus | None = None,
    ) -> list[DocumentRecord]:
        query = "SELECT * FROM documents WHERE 1 = 1"
        parameters: list[str] = []

        if scope is not None:
            query += " AND scope = ?"
            parameters.append(scope.value)

        if conversation_id is not None:
            query += " AND conversation_id = ?"
            parameters.append(str(conversation_id))

        if status is not None:
            query += " AND status = ?"
            parameters.append(status.value)

        query += " ORDER BY created_at DESC"

        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()

        return [self._document_from_row(row) for row in rows]

    def update_document_status(
        self,
        document_id: UUID,
        status: DocumentStatus,
        *,
        error_message: str | None = None,
        page_count: int | None = None,
        chunk_count: int | None = None,
    ) -> bool:
        now = self._serialize_datetime(self._utc_now())

        with self._connect() as connection:
            result = connection.execute(
                """
                UPDATE documents
                SET
                    status = ?,
                    error_message = ?,
                    page_count = COALESCE(?, page_count),
                    chunk_count = COALESCE(?, chunk_count),
                    updated_at = ?,
                    indexed_at = CASE
                        WHEN ? = 'indexed' THEN ?
                        ELSE indexed_at
                    END
                WHERE id = ?
                """,
                (
                    status.value,
                    error_message,
                    page_count,
                    chunk_count,
                    now,
                    status.value,
                    now,
                    str(document_id),
                ),
            )

        return result.rowcount > 0

    def add_chunks(self, chunks: Sequence[DocumentChunk]) -> None:
        if not chunks:
            return

        now_datetime = self._utc_now()
        now = self._serialize_datetime(now_datetime)

        chunk_counts: dict[UUID, int] = defaultdict(int)

        for chunk in chunks:
            self._validate_datetime(chunk.created_at)
            chunk_counts[chunk.document_id] += 1

        with self._connect() as connection:
            connection.executemany(
                """
                INSERT INTO document_chunks (
                    id,
                    document_id,
                    chunk_index,
                    content,
                    page_number,
                    start_char,
                    end_char,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        str(chunk.id),
                        str(chunk.document_id),
                        chunk.chunk_index,
                        chunk.content,
                        chunk.page_number,
                        None,
                        None,
                        (
                            self._serialize_datetime(chunk.created_at)
                            or now
                        ),
                    )
                    for chunk in chunks
                ],
            )

            for document_id, count in chunk_counts.items():
                connection.execute(
                    """
                    UPDATE documents
                    SET
                        chunk_count = chunk_count + ?,
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (count, now, str(document_id)),
                )

    def list_chunks(self, document_id: UUID) -> list[DocumentChunk]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM document_chunks
                WHERE document_id = ?
                ORDER BY chunk_index ASC
                """,
                (str(document_id),),
            ).fetchall()

        return [self._chunk_from_row(row) for row in rows]

    def list_embeddings(
        self,
        document_id: UUID,
    ) -> list[DocumentEmbedding]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT e.*
                FROM chunk_embeddings AS e
                INNER JOIN document_chunks AS c
                    ON c.id = e.chunk_id
                WHERE c.document_id = ?
                ORDER BY c.chunk_index ASC
                """,
                (str(document_id),),
            ).fetchall()

        return [self._embedding_from_row(row) for row in rows]

    def replace_index(
        self,
        document_id: UUID,
        chunks: Sequence[DocumentChunk],
        embeddings: Sequence[DocumentEmbedding],
        *,
        page_count: int,
    ) -> None:
        """Atomically replace a document's chunks and embeddings."""
        if page_count < 0:
            raise ValueError("Page count cannot be negative.")

        chunk_ids = [chunk.id for chunk in chunks]

        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("Duplicate chunk IDs are not allowed.")

        for chunk in chunks:
            if chunk.document_id != document_id:
                raise ValueError(
                    "Every chunk must belong to the target document."
                )

            if chunk.chunk_index < 0:
                raise ValueError("Chunk indexes cannot be negative.")

            if not chunk.content.strip():
                raise ValueError("Chunk content cannot be empty.")

            self._validate_datetime(chunk.created_at)

        embedding_ids = [embedding.chunk_id for embedding in embeddings]

        if len(embedding_ids) != len(set(embedding_ids)):
            raise ValueError("Duplicate chunk embeddings are not allowed.")

        if set(embedding_ids) != set(chunk_ids):
            raise ValueError(
                "Every chunk must have exactly one matching embedding."
            )

        encoded_embeddings = {
            embedding.chunk_id: self._validate_and_encode_embedding(
                embedding
            )
            for embedding in embeddings
        }

        now = self._serialize_datetime(self._utc_now())

        with self._connect() as connection:
            document_exists = connection.execute(
                "SELECT 1 FROM documents WHERE id = ?",
                (str(document_id),),
            ).fetchone()

            if document_exists is None:
                raise ValueError(
                    "Cannot index a document that does not exist."
                )

            connection.execute(
                """
                DELETE FROM document_chunks
                WHERE document_id = ?
                """,
                (str(document_id),),
            )

            connection.executemany(
                """
                INSERT INTO document_chunks (
                    id,
                    document_id,
                    chunk_index,
                    content,
                    page_number,
                    start_char,
                    end_char,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        str(chunk.id),
                        str(chunk.document_id),
                        chunk.chunk_index,
                        chunk.content,
                        chunk.page_number,
                        None,
                        None,
                        (
                            self._serialize_datetime(chunk.created_at)
                            or now
                        ),
                    )
                    for chunk in chunks
                ],
            )

            connection.executemany(
                """
                INSERT INTO chunk_embeddings (
                    chunk_id,
                    model_name,
                    dimensions,
                    vector,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (
                        str(embedding.chunk_id),
                        embedding.model_name,
                        embedding.dimensions,
                        encoded_embeddings[embedding.chunk_id],
                        (
                            self._serialize_datetime(embedding.created_at)
                            or now
                        ),
                    )
                    for embedding in embeddings
                ],
            )

            connection.execute(
                """
                UPDATE documents
                SET
                    status = ?,
                    page_count = ?,
                    chunk_count = ?,
                    error_message = NULL,
                    indexed_at = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    DocumentStatus.INDEXED.value,
                    page_count,
                    len(chunks),
                    now,
                    now,
                    str(document_id),
                ),
            )

    def delete_document(self, document_id: UUID) -> bool:
        with self._connect() as connection:
            result = connection.execute(
                "DELETE FROM documents WHERE id = ?",
                (str(document_id),),
            )

        return result.rowcount > 0

    def delete_conversation_documents(
        self,
        conversation_id: UUID,
    ) -> int:
        with self._connect() as connection:
            result = connection.execute(
                """
                DELETE FROM documents
                WHERE scope = ?
                    AND conversation_id = ?
                """,
                (
                    DocumentScope.TEMPORARY.value,
                    str(conversation_id),
                ),
            )

        return result.rowcount

    def touch_conversation_documents(
        self,
        conversation_id: UUID,
        *,
        retention_hours: int = 24,
    ) -> int:
        if retention_hours <= 0:
            raise ValueError("Retention hours must be greater than zero.")

        now_datetime = self._utc_now()
        now = self._serialize_datetime(now_datetime)
        expires_at = self._serialize_datetime(
            now_datetime + timedelta(hours=retention_hours)
        )

        with self._connect() as connection:
            result = connection.execute(
                """
                UPDATE documents
                SET
                    last_accessed_at = ?,
                    updated_at = ?,
                    expires_at = ?
                WHERE scope = ?
                    AND conversation_id = ?
                """,
                (
                    now,
                    now,
                    expires_at,
                    DocumentScope.TEMPORARY.value,
                    str(conversation_id),
                ),
            )

        return result.rowcount

    def delete_expired_documents(
        self,
        *,
        now: datetime | None = None,
    ) -> int:
        now_datetime = now or self._utc_now()
        self._validate_datetime(now_datetime)
        serialized_now = self._serialize_datetime(now_datetime)

        with self._connect() as connection:
            result = connection.execute(
                """
                DELETE FROM documents
                WHERE scope = ?
                    AND expires_at IS NOT NULL
                    AND expires_at <= ?
                """,
                (
                    DocumentScope.TEMPORARY.value,
                    serialized_now,
                ),
            )

        return result.rowcount

    def search_chunks(
        self,
        query_vector: Sequence[float],
        *,
        model_name: str,
        dimensions: int,
        knowledge_scope: str,
        conversation_id: UUID | None = None,
        limit: int = 5,
        now: datetime | None = None,
    ) -> list[dict[str, object]]:
        """Return the most similar chunks from the requested knowledge scope."""
        if knowledge_scope not in {"none", "conversation", "library", "both"}:
            raise ValueError("Unsupported knowledge scope.")

        if knowledge_scope == "none":
            return []

        if knowledge_scope in {"conversation", "both"} and conversation_id is None:
            raise ValueError(
                "A conversation ID is required for this knowledge scope."
            )

        if not model_name or not model_name.strip():
            raise ValueError("Embedding model name cannot be empty.")

        if dimensions <= 0 or len(query_vector) != dimensions:
            raise ValueError(
                "Query embedding dimensions do not match the expected dimensions."
            )

        if limit <= 0:
            raise ValueError("Search limit must be greater than zero.")

        try:
            vector = [float(value) for value in query_vector]
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Query embedding contains invalid values.") from exc

        if not all(math.isfinite(value) for value in vector):
            raise ValueError("Query embedding must contain finite values.")

        query_norm = math.sqrt(sum(value * value for value in vector))
        if query_norm == 0:
            raise ValueError("Query embedding cannot be a zero vector.")

        now_datetime = now or self._utc_now()
        self._validate_datetime(now_datetime)
        serialized_now = self._serialize_datetime(now_datetime)

        conditions = [
            "d.status = ?",
            "e.model_name = ?",
            "e.dimensions = ?",
        ]
        parameters: list[object] = [
            DocumentStatus.INDEXED.value,
            model_name,
            dimensions,
        ]

        if knowledge_scope == "conversation":
            conditions.extend([
                "d.scope = ?",
                "d.conversation_id = ?",
                "d.expires_at IS NOT NULL",
                "d.expires_at > ?",
            ])
            parameters.extend([
                DocumentScope.TEMPORARY.value,
                str(conversation_id),
                serialized_now,
            ])

        elif knowledge_scope == "library":
            conditions.extend([
                "d.scope = ?",
                "d.conversation_id IS NULL",
            ])
            parameters.append(DocumentScope.PERSISTENT.value)

        else:  # both
            conditions.append(
                """
                (
                    (
                        d.scope = ?
                        AND d.conversation_id IS NULL
                    )
                    OR
                    (
                        d.scope = ?
                        AND d.conversation_id = ?
                        AND d.expires_at IS NOT NULL
                        AND d.expires_at > ?
                    )
                )
                """
            )
            parameters.extend([
                DocumentScope.PERSISTENT.value,
                DocumentScope.TEMPORARY.value,
                str(conversation_id),
                serialized_now,
            ])

        query = f"""
            SELECT
                d.id AS document_id,
                d.filename,
                d.scope,
                c.id AS chunk_id,
                c.chunk_index,
                c.content,
                c.page_number,
                e.model_name,
                e.dimensions,
                e.vector,
                e.created_at
            FROM documents AS d
            INNER JOIN document_chunks AS c
                ON c.document_id = d.id
            INNER JOIN chunk_embeddings AS e
                ON e.chunk_id = c.id
            WHERE {' AND '.join(conditions)}
        """

        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()

        matches: list[dict[str, object]] = []

        for row in rows:
            try:
                embedding = self._embedding_from_row(row)
            except (ValueError, struct.error):
                # Ignore corrupt vectors rather than failing the entire search.
                continue

            stored_vector = embedding.vector
            stored_norm = math.sqrt(
                sum(value * value for value in stored_vector)
            )

            if stored_norm == 0:
                continue

            dot_product = sum(
                query_value * stored_value
                for query_value, stored_value in zip(
                    vector,
                    stored_vector,
                )
            )

            similarity = dot_product / (query_norm * stored_norm)

            matches.append({
                "document_id": UUID(row["document_id"]),
                "filename": row["filename"],
                "scope": DocumentScope(row["scope"]),
                "chunk_id": UUID(row["chunk_id"]),
                "chunk_index": row["chunk_index"],
                "content": row["content"],
                "page_number": row["page_number"],
                "similarity": similarity,
            })

        matches.sort(
            key=lambda item: (
                float(item["similarity"]),
                -int(item["chunk_index"]),
            ),
            reverse=True,
        )

        return matches[:limit]
