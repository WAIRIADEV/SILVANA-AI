
import fnmatch
import os

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.app.intelligence.filesystem.service import FilesystemService
from backend.app.intelligence.tool_registry import (
    ToolDefinition,
    ToolRegistry,
)
from backend.app.intelligence.verification_engine import (
    VerificationEngine,
)


def _object_schema(
    properties: dict[str, Any],
    required: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
        "additionalProperties": False,
    }


def _path_schema() -> dict[str, Any]:
    return {
        "type": "string",
        "minLength": 1,
        "maxLength": 1024,
    }


def _verify_list_directory(
    service: FilesystemService,
    result: Any,
) -> bool:
    if not isinstance(result, dict):
        return False

    if result.get("operation") != "list_directory":
        return False

    path = result.get("path")
    entries = result.get("entries")

    if not isinstance(path, str) or not isinstance(entries, list):
        return False

    if result.get("count") != len(entries):
        return False

    if not isinstance(result.get("truncated"), bool):
        return False

    directory = service.boundary.resolve(path)

    if not directory.is_dir():
        return False

    actual_entries = {}

    with os.scandir(directory) as iterator:
        for entry in iterator:
            if entry.is_symlink():
                kind = "symlink"
            elif entry.is_dir(follow_symlinks=False):
                kind = "directory"
            elif entry.is_file(follow_symlinks=False):
                kind = "file"
            else:
                kind = "other"

            actual_entries[entry.name] = kind

    reported = {}

    for item in entries:
        if not isinstance(item, dict):
            return False

        name = item.get("name")
        kind = item.get("kind")

        if not isinstance(name, str):
            return False

        if name in reported:
            return False

        if actual_entries.get(name) != kind:
            return False

        reported[name] = kind

    if not result["truncated"]:
        return reported == actual_entries

    return len(reported) <= len(actual_entries)


def _verify_search_files(
    service: FilesystemService,
    result: Any,
) -> bool:
    if not isinstance(result, dict):
        return False

    if result.get("operation") != "search_files":
        return False

    directory_path = result.get("directory")
    pattern = result.get("pattern")
    results = result.get("results")

    if not isinstance(directory_path, str):
        return False

    if not isinstance(pattern, str):
        return False

    if not isinstance(results, list):
        return False

    if result.get("count") != len(results):
        return False

    truncated = result.get("truncated")
    scan_limited = result.get("scan_limited")
    scanned_entries = result.get("scanned_entries")

    if not isinstance(truncated, bool):
        return False

    if not isinstance(scan_limited, bool):
        return False

    if (
        isinstance(scanned_entries, bool)
        or not isinstance(scanned_entries, int)
        or scanned_entries < 0
        or scanned_entries > service.MAX_SEARCH_ENTRIES
    ):
        return False

    if scan_limited and scanned_entries != service.MAX_SEARCH_ENTRIES:
        return False

    directory = service.boundary.resolve(directory_path)

    if not directory.is_dir():
        return False

    seen_paths = set()

    for item in results:
        if not isinstance(item, dict):
            return False

        relative_path = item.get("path")
        name = item.get("name")

        if not isinstance(relative_path, str):
            return False

        if not isinstance(name, str):
            return False

        if relative_path in seen_paths:
            return False

        seen_paths.add(relative_path)

        target = service.boundary.resolve(relative_path)

        try:
            target.relative_to(directory)
        except ValueError:
            return False

        if target.name != name:
            return False

        if not fnmatch.fnmatch(name.lower(), pattern.lower()):
            return False

        if not target.is_file():
            return False

    return True


def _verify_file_metadata(
    service: FilesystemService,
    result: Any,
) -> bool:
    if not isinstance(result, dict):
        return False

    if result.get("operation") != "get_file_metadata":
        return False

    path = result.get("path")

    if not isinstance(path, str):
        return False

    target = service.boundary.resolve(path)

    if not target.is_file() and not target.is_dir():
        return False

    stat = target.stat()

    expected_kind = (
        "directory" if target.is_dir() else "file"
    )

    expected_extension = (
        target.suffix.lower()
        if target.is_file()
        else None
    )

    expected_modified = datetime.fromtimestamp(
        stat.st_mtime,
        tz=timezone.utc,
    ).isoformat()

    return (
        result.get("kind") == expected_kind
        and result.get("size_bytes") == stat.st_size
        and result.get("extension") == expected_extension
        and result.get("modified_at_utc") == expected_modified
    )


def _verify_file_exists(
    service: FilesystemService,
    result: Any,
) -> bool:
    if not isinstance(result, dict):
        return False

    if result.get("operation") != "check_file_exists":
        return False

    path = result.get("path")

    if not isinstance(path, str):
        return False

    target = service.boundary.resolve(
        path,
        must_exist=False,
    )

    exists = target.exists()

    if result.get("exists") is not exists:
        return False

    if not exists:
        return result.get("kind") == "missing"

    if target.is_file():
        expected_kind = "file"
    elif target.is_dir():
        expected_kind = "directory"
    else:
        expected_kind = "other"

    return result.get("kind") == expected_kind


def register_filesystem_tools(
    registry: ToolRegistry,
    verification_engine: VerificationEngine,
    service: FilesystemService,
) -> None:
    """Register bounded, read-only filesystem capabilities."""

    intent = {"tool_action"}

    tools = [
        ToolDefinition(
            name="list_directory",
            description=(
                "List files and directories inside the permitted workspace."
            ),
            handler=service.list_directory,
            parameters=_object_schema({
                "path": _path_schema(),
                "max_entries": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 200,
                },
            }),
            permission_level="read",
            supported_intents=intent,
        ),
        ToolDefinition(
            name="search_files",
            description=(
                "Search workspace filenames using a bounded filename pattern."
            ),
            handler=service.search_files,
            parameters=_object_schema({
                "directory": _path_schema(),
                "pattern": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 255,
                },
                "recursive": {
                    "type": "boolean",
                },
                "max_results": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                },
                "max_depth": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10,
                },
            }),
            permission_level="read",
            supported_intents=intent,
        ),
        ToolDefinition(
            name="get_file_metadata",
            description=(
                "Inspect file or directory metadata without reading file contents."
            ),
            handler=service.get_file_metadata,
            parameters=_object_schema(
                {"path": _path_schema()},
                required=["path"],
            ),
            permission_level="read",
            supported_intents=intent,
        ),
        ToolDefinition(
            name="check_file_exists",
            description=(
                "Check whether a workspace-relative path exists."
            ),
            handler=service.check_file_exists,
            parameters=_object_schema(
                {"path": _path_schema()},
                required=["path"],
            ),
            permission_level="read",
            supported_intents=intent,
        ),
    ]

    verifiers = {
        "list_directory": lambda result: (
            _verify_list_directory(service, result)
        ),
        "search_files": lambda result: (
            _verify_search_files(service, result)
        ),
        "get_file_metadata": lambda result: (
            _verify_file_metadata(service, result)
        ),
        "check_file_exists": lambda result: (
            _verify_file_exists(service, result)
        ),
    }

    for tool in tools:
        registry.register(tool)

    for tool_name, verifier in verifiers.items():
        verification_engine.register(tool_name, verifier)
