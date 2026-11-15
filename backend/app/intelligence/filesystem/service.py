
import fnmatch
import os

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.app.intelligence.filesystem.workspace import (
    WorkspaceBoundary,
)


class FilesystemService:
    """Provides bounded, read-only access to the configured workspace."""

    MAX_DIRECTORY_ENTRIES = 200
    MAX_SEARCH_RESULTS = 100
    MAX_SEARCH_DEPTH = 10
    MAX_SEARCH_ENTRIES = 5000

    def __init__(self, boundary: WorkspaceBoundary):
        self.boundary = boundary

    def _relative_path(self, path: Path) -> str:
        return path.relative_to(self.boundary.root).as_posix() or "."

    @staticmethod
    def _validate_limit(
        value: int,
        maximum: int,
        name: str,
    ) -> None:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer.")

        if value < 1 or value > maximum:
            raise ValueError(
                f"{name} must be between 1 and {maximum}."
            )

    def list_directory(
        self,
        path: str = ".",
        max_entries: int = 100,
    ) -> dict[str, Any]:
        self._validate_limit(
            max_entries,
            self.MAX_DIRECTORY_ENTRIES,
            "max_entries",
        )

        directory = self.boundary.resolve(path)

        if not directory.is_dir():
            raise ValueError("The requested path is not a directory.")

        entries = []
        truncated = False

        with os.scandir(directory) as iterator:
            for entry in iterator:
                if len(entries) >= max_entries:
                    truncated = True
                    break

                if entry.is_symlink():
                    kind = "symlink"
                elif entry.is_dir(follow_symlinks=False):
                    kind = "directory"
                elif entry.is_file(follow_symlinks=False):
                    kind = "file"
                else:
                    kind = "other"

                entries.append({
                    "name": entry.name,
                    "kind": kind,
                })

        entries.sort(key=lambda item: item["name"].lower())

        return {
            "operation": "list_directory",
            "path": self._relative_path(directory),
            "entries": entries,
            "count": len(entries),
            "truncated": truncated,
        }

    def search_files(
        self,
        directory: str = ".",
        pattern: str = "*",
        recursive: bool = True,
        max_results: int = 50,
        max_depth: int = 5,
    ) -> dict[str, Any]:
        self._validate_limit(
            max_results,
            self.MAX_SEARCH_RESULTS,
            "max_results",
        )

        self._validate_limit(
            max_depth,
            self.MAX_SEARCH_DEPTH,
            "max_depth",
        )

        if not isinstance(pattern, str) or not pattern.strip():
            raise ValueError("A non-empty filename pattern is required.")

        if len(pattern) > 255:
            raise ValueError("The filename pattern is too long.")

        if not isinstance(recursive, bool):
            raise ValueError("recursive must be a boolean.")

        start = self.boundary.resolve(directory)

        if not start.is_dir():
            raise ValueError("The search location is not a directory.")

        results = []
        truncated = False
        scan_limited = False
        scanned_entries = 0

        directories = [(start, 0)]

        while directories:
            current, depth = directories.pop()

            current = self.boundary.resolve(
                self._relative_path(current)
            )

            with os.scandir(current) as iterator:
                for entry in iterator:
                    if scanned_entries >= self.MAX_SEARCH_ENTRIES:
                        scan_limited = True
                        break

                    scanned_entries += 1

                    if entry.is_symlink():
                        continue

                    if entry.is_dir(follow_symlinks=False):
                        if recursive and depth < max_depth:
                            candidate = Path(entry.path)

                            # Validate discovered directories before
                            # adding them to the traversal queue.
                            candidate = self.boundary.resolve(
                                self._relative_path(candidate)
                            )

                            directories.append(
                                (candidate, depth + 1)
                            )

                        continue

                    if not entry.is_file(follow_symlinks=False):
                        continue

                    if not fnmatch.fnmatch(
                        entry.name.lower(),
                        pattern.lower(),
                    ):
                        continue

                    # Only an additional matching file proves that
                    # the result limit omitted a matching result.
                    if len(results) >= max_results:
                        truncated = True
                        break

                    result_path = Path(entry.path)

                    results.append({
                        "name": entry.name,
                        "path": self._relative_path(result_path),
                    })

                if truncated or scan_limited:
                    break

        results.sort(key=lambda item: item["path"].lower())

        return {
            "operation": "search_files",
            "directory": self._relative_path(start),
            "pattern": pattern,
            "results": results,
            "count": len(results),
            "truncated": truncated,
            "scan_limited": scan_limited,
            "scanned_entries": scanned_entries,
        }

    def get_file_metadata(self, path: str) -> dict[str, Any]:
        target = self.boundary.resolve(path)

        if not target.is_file() and not target.is_dir():
            raise ValueError(
                "Metadata is available only for files and directories."
            )

        metadata = target.stat()

        modified = datetime.fromtimestamp(
            metadata.st_mtime,
            tz=timezone.utc,
        ).isoformat()

        return {
            "operation": "get_file_metadata",
            "path": self._relative_path(target),
            "kind": "directory" if target.is_dir() else "file",
            "size_bytes": metadata.st_size,
            "extension": (
                target.suffix.lower() if target.is_file() else None
            ),
            "modified_at_utc": modified,
        }

    def check_file_exists(self, path: str) -> dict[str, Any]:
        target = self.boundary.resolve(
            path,
            must_exist=False,
        )

        exists = target.exists()

        if not exists:
            kind = "missing"
        elif target.is_file():
            kind = "file"
        elif target.is_dir():
            kind = "directory"
        else:
            kind = "other"

        return {
            "operation": "check_file_exists",
            "path": self._relative_path(target),
            "exists": exists,
            "kind": kind,
        }
