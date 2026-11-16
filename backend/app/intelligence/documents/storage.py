
import os
from pathlib import Path
from uuid import UUID, uuid4


class DocumentStorage:
    """Safely stores uploaded documents outside the user workspace."""

    def __init__(
        self,
        storage_root: Path,
        *,
        max_file_size_bytes: int = 20 * 1024 * 1024,
    ) -> None:
        if max_file_size_bytes < 1:
            raise ValueError("Maximum file size must be positive.")

        self.storage_root = Path(storage_root).resolve()
        self.max_file_size_bytes = max_file_size_bytes

        self.storage_root.mkdir(
            parents=True,
            exist_ok=True,
        )

    def _resolve_storage_path(self, storage_key: str) -> Path:
        """Resolve a storage key while preventing directory traversal."""

        if not storage_key or Path(storage_key).name != storage_key:
            raise ValueError("Invalid storage key.")

        if storage_key in {".", ".."}:
            raise ValueError("Invalid storage key.")

        candidate = (self.storage_root / storage_key).resolve()

        if candidate.parent != self.storage_root:
            raise ValueError("Storage path escapes the storage directory.")

        return candidate

    def save(
        self,
        content: bytes,
        *,
        document_id: UUID | None = None,
    ) -> str:
        """Atomically store document bytes and return their storage key."""

        if not isinstance(content, bytes):
            raise TypeError("Document content must be bytes.")

        if not content:
            raise ValueError("Document content cannot be empty.")

        if len(content) > self.max_file_size_bytes:
            raise ValueError("Document exceeds the maximum allowed size.")

        identifier = document_id or uuid4()
        storage_key = f"{identifier.hex}.document"
        destination = self._resolve_storage_path(storage_key)

        if destination.exists():
            raise FileExistsError("Document storage key already exists.")

        temporary_path = self._resolve_storage_path(
            f".{uuid4().hex}.tmp"
        )

        try:
            with temporary_path.open("xb") as file:
                file.write(content)
                file.flush()
                os.fsync(file.fileno())

            # Atomic creation without overwriting an existing destination.
            os.link(temporary_path, destination)

        finally:
            temporary_path.unlink(missing_ok=True)

        return storage_key

    def read(self, storage_key: str) -> bytes:
        """Read a stored document."""

        path = self._resolve_storage_path(storage_key)

        if not path.is_file():
            raise FileNotFoundError("Stored document does not exist.")

        return path.read_bytes()

    def delete(self, storage_key: str) -> bool:
        """Delete a stored document if it exists."""

        path = self._resolve_storage_path(storage_key)

        if not path.exists():
            return False

        if not path.is_file():
            raise ValueError("Storage target is not a regular file.")

        path.unlink()
        return True

    def exists(self, storage_key: str) -> bool:
        """Check whether a stored document exists."""

        return self._resolve_storage_path(storage_key).is_file()
