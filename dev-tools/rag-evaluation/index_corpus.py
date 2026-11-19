
import hashlib
import mimetypes
import sys
from pathlib import Path
from uuid import uuid4


# Ensure the repository root is importable when this script
# is executed directly from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from backend.app.intelligence.documents.chunker import DocumentChunker
from backend.app.intelligence.documents.extractor import DocumentExtractor
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
from backend.app.intelligence.embeddings.fastembed import FastEmbedProvider

from config import evaluation_config


def build_indexing_service():
    """Construct the production indexing pipeline in isolated storage."""

    evaluation_config.prepare_directories()

    repository = DocumentRepository(
        database_path=evaluation_config.database_path
    )
    repository.initialize()

    storage = DocumentStorage(
        storage_root=evaluation_config.storage_root,
        max_file_size_bytes=evaluation_config.max_file_size_bytes,
    )

    embedding_provider = FastEmbedProvider(
        model_name=evaluation_config.embedding_model,
        batch_size=evaluation_config.embedding_batch_size,
        threads=evaluation_config.embedding_threads,
    )

    extractor = DocumentExtractor()

    chunker = DocumentChunker(
        token_counter=embedding_provider.count_tokens,
        max_tokens=evaluation_config.chunk_max_tokens,
        token_overlap=evaluation_config.chunk_token_overlap,
    )

    indexing_service = DocumentIndexingService(
        repository=repository,
        storage=storage,
        extractor=extractor,
        chunker=chunker,
        embedding_provider=embedding_provider,
    )

    return repository, storage, indexing_service


def discover_corpus_files() -> list[Path]:
    """Find supported documents without following paths outside the corpus."""

    corpus_root = evaluation_config.corpus_root.resolve()

    if not corpus_root.is_dir():
        raise FileNotFoundError(
            f"Evaluation corpus directory does not exist: {corpus_root}"
        )

    files = []

    for path in sorted(corpus_root.rglob("*")):
        if not path.is_file():
            continue

        resolved_path = path.resolve()

        if not resolved_path.is_relative_to(corpus_root):
            raise RuntimeError(
                f"Corpus file escapes the corpus directory: {path}"
            )

        if path.suffix.lower() not in DocumentExtractor.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported corpus file format: {path.name}"
            )

        files.append(path)

    if not files:
        raise RuntimeError(
            "No supported documents were found in the evaluation corpus."
        )

    return files


def index_corpus() -> dict[str, int]:
    """Register and index corpus documents without destructive replacement."""

    repository, storage, indexing_service = build_indexing_service()

    existing_documents = repository.list_documents(
        scope=DocumentScope.PERSISTENT,
    )

    existing_by_filename = {}

    for document in existing_documents:
        existing_by_filename.setdefault(
            document.filename,
            [],
        ).append(document)

    files = discover_corpus_files()

    summary = {
        "discovered": len(files),
        "indexed": 0,
        "skipped": 0,
        "retried": 0,
        "failed": 0,
    }

    corpus_root = evaluation_config.corpus_root.resolve()

    for file_path in files:
        relative_filename = file_path.resolve().relative_to(
            corpus_root
        ).as_posix()

        content = file_path.read_bytes()

        if not content:
            raise ValueError(
                f"Corpus document is empty: {relative_filename}"
            )

        if len(content) > evaluation_config.max_file_size_bytes:
            raise ValueError(
                f"Corpus document exceeds the configured size limit: "
                f"{relative_filename}"
            )

        content_hash = hashlib.sha256(content).hexdigest()

        matching_documents = existing_by_filename.get(
            relative_filename,
            [],
        )

        if len(matching_documents) > 1:
            raise RuntimeError(
                f"Multiple evaluation records exist for "
                f"{relative_filename}. Resolve the duplicate manually."
            )

        if matching_documents:
            existing = matching_documents[0]

            if existing.content_hash != content_hash:
                raise RuntimeError(
                    f"Corpus content changed for {relative_filename}. "
                    "Automatic replacement is disabled. "
                    "Resolve this change explicitly."
                )

            if existing.status == DocumentStatus.INDEXED:
                print(f"SKIPPED  {relative_filename}")
                summary["skipped"] += 1
                continue

            print(f"RETRYING {relative_filename}")
            summary["retried"] += 1

            try:
                indexed = indexing_service.index_document(existing.id)

                if indexed.status != DocumentStatus.INDEXED:
                    raise DocumentIndexingError(
                        "Document did not reach INDEXED status."
                    )

                print(
                    f"INDEXED  {relative_filename} "
                    f"({indexed.chunk_count} chunks)"
                )
                summary["indexed"] += 1

            except Exception as exc:
                print(f"FAILED   {relative_filename}: {exc}")
                summary["failed"] += 1

            continue

        mime_type, _ = mimetypes.guess_type(relative_filename)

        document_id = uuid4()

        storage_key = storage.save(
            content,
            document_id=document_id,
        )

        document = DocumentRecord(
            id=document_id,
            scope=DocumentScope.PERSISTENT,
            conversation_id=None,
            filename=relative_filename,
            storage_key=storage_key,
            content_hash=content_hash,
            mime_type=mime_type or "application/octet-stream",
            size_bytes=len(content),
            status=DocumentStatus.UPLOADED,
        )

        try:
            repository.create_document(document)

        except Exception:
            storage.delete(storage_key)
            raise

        try:
            indexed = indexing_service.index_document(document_id)

            if indexed.status != DocumentStatus.INDEXED:
                raise DocumentIndexingError(
                    "Document did not reach INDEXED status."
                )

            print(
                f"INDEXED  {relative_filename} "
                f"({indexed.chunk_count} chunks)"
            )
            summary["indexed"] += 1

        except Exception as exc:
            # Keep the failed record and its source file for diagnosis
            # and a possible retry on the next run.
            print(f"FAILED   {relative_filename}: {exc}")
            summary["failed"] += 1

    return summary


def main() -> int:
    print("SILVANA RAG EVALUATION — CORPUS INDEXING")
    print(f"Database: {evaluation_config.database_path}")
    print(f"Storage:  {evaluation_config.storage_root}")
    print(f"Corpus:   {evaluation_config.corpus_root}")
    print()

    summary = index_corpus()

    print()
    print("INDEXING SUMMARY")

    for key, value in summary.items():
        print(f"{key.capitalize():12} {value}")

    if summary["failed"]:
        print()
        print("Some documents failed indexing. Inspect the errors above.")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
