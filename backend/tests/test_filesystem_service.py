
import pytest

from backend.app.intelligence.filesystem.workspace import WorkspaceBoundary
from backend.app.intelligence.filesystem.service import FilesystemService


@pytest.fixture
def filesystem(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()

    documents = root / "Documents"
    documents.mkdir()

    reports = documents / "Reports"
    reports.mkdir()

    (documents / "notes.txt").write_text(
        "Private test content",
        encoding="utf-8",
    )
    (reports / "annual_report.pdf").write_bytes(b"PDF test data")
    (root / "readme.md").write_text(
        "SILVANA filesystem tests",
        encoding="utf-8",
    )

    boundary = WorkspaceBoundary(root)
    return FilesystemService(boundary), root


# DIRECTORY LISTING

def test_lists_directory_entries(filesystem):
    service, _ = filesystem

    result = service.list_directory("Documents")

    assert result["operation"] == "list_directory"
    assert result["path"] == "Documents"
    assert result["count"] == 2

    names = {entry["name"] for entry in result["entries"]}
    assert names == {"notes.txt", "Reports"}


def test_directory_listing_respects_entry_limit(filesystem):
    service, root = filesystem

    for index in range(5):
        (root / f"file_{index}.txt").touch()

    result = service.list_directory(".", max_entries=2)

    assert result["count"] == 2
    assert result["truncated"] is True


def test_directory_listing_rejects_file_path(filesystem):
    service, _ = filesystem

    with pytest.raises(ValueError):
        service.list_directory("readme.md")


# FILE SEARCH

def test_searches_files_by_filename(filesystem):
    service, _ = filesystem

    result = service.search_files(
        pattern="*.txt",
        recursive=True,
    )

    paths = {item["path"] for item in result["results"]}

    assert result["count"] == 1
    assert "Documents/notes.txt" in paths


def test_searches_recursively(filesystem):
    service, _ = filesystem

    result = service.search_files(
        pattern="*.pdf",
        recursive=True,
    )

    assert result["count"] == 1
    assert result["results"][0]["path"] == (
        "Documents/Reports/annual_report.pdf"
    )


def test_non_recursive_search_does_not_enter_subdirectories(filesystem):
    service, _ = filesystem

    result = service.search_files(
        pattern="*.pdf",
        recursive=False,
    )

    assert result["count"] == 0


def test_search_respects_result_limit(filesystem):
    service, root = filesystem

    for index in range(5):
        (root / f"report_{index}.txt").touch()

    result = service.search_files(
        pattern="*.txt",
        max_results=2,
    )

    assert result["count"] == 2
    assert result["truncated"] is True


def test_search_exact_result_limit_is_not_marked_truncated(filesystem):
    service, root = filesystem

    for index in range(3):
        (root / f"exact_{index}.txt").touch()

    result = service.search_files(
        pattern="exact_*.txt",
        max_results=3,
    )

    assert result["count"] == 3
    assert result["truncated"] is False
    assert result["scan_limited"] is False


def test_search_reports_when_scan_budget_is_exhausted(filesystem):
    service, root = filesystem

    for index in range(service.MAX_SEARCH_ENTRIES + 1):
        (root / f"entry_{index}.bin").touch()

    result = service.search_files(
        pattern="*.txt",
        max_results=10,
    )

    assert result["scan_limited"] is True
    assert result["scanned_entries"] == service.MAX_SEARCH_ENTRIES


def test_search_rejects_invalid_directory(filesystem):
    service, _ = filesystem

    with pytest.raises(ValueError):
        service.search_files(directory="readme.md")


def test_search_rejects_empty_pattern(filesystem):
    service, _ = filesystem

    with pytest.raises(ValueError):
        service.search_files(pattern=" ")


def test_search_rejects_excessive_depth(filesystem):
    service, _ = filesystem

    with pytest.raises(ValueError):
        service.search_files(max_depth=100)


def test_search_rejects_excessive_result_limit(filesystem):
    service, _ = filesystem

    with pytest.raises(ValueError):
        service.search_files(max_results=1000)


# FILE METADATA

def test_reads_file_metadata(filesystem):
    service, _ = filesystem

    result = service.get_file_metadata("Documents/notes.txt")

    assert result["operation"] == "get_file_metadata"
    assert result["kind"] == "file"
    assert result["extension"] == ".txt"
    assert result["size_bytes"] == len("Private test content")


def test_reads_directory_metadata(filesystem):
    service, _ = filesystem

    result = service.get_file_metadata("Documents")

    assert result["kind"] == "directory"
    assert result["extension"] is None


def test_metadata_rejects_missing_path(filesystem):
    service, _ = filesystem

    with pytest.raises(Exception):
        service.get_file_metadata("missing.txt")


# EXISTENCE CHECK

def test_confirms_existing_file(filesystem):
    service, _ = filesystem

    result = service.check_file_exists("readme.md")

    assert result["exists"] is True
    assert result["kind"] == "file"


def test_confirms_existing_directory(filesystem):
    service, _ = filesystem

    result = service.check_file_exists("Documents")

    assert result["exists"] is True
    assert result["kind"] == "directory"


def test_reports_missing_path(filesystem):
    service, _ = filesystem

    result = service.check_file_exists("missing.txt")

    assert result["exists"] is False
    assert result["kind"] == "missing"


# SECURITY

@pytest.mark.parametrize(
    "path",
    [
        "../outside.txt",
        "../../secret.txt",
        "C:\\Windows\\win.ini",
        "/etc/passwd",
    ],
)
def test_operations_reject_paths_outside_workspace(filesystem, path):
    service, _ = filesystem

    with pytest.raises(Exception):
        service.check_file_exists(path)


def test_search_does_not_follow_directory_symlinks(filesystem, tmp_path):
    service, root = filesystem

    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text(
        "Must not be discovered",
        encoding="utf-8",
    )

    link = root / "external"
    link.symlink_to(outside, target_is_directory=True)

    result = service.search_files(
        pattern="*.txt",
        recursive=True,
    )

    paths = {item["path"] for item in result["results"]}

    assert not any(path.startswith("external/") for path in paths)


def test_listing_identifies_symlinks_without_following_them(filesystem, tmp_path):
    service, root = filesystem

    outside = tmp_path / "outside"
    outside.mkdir()

    link = root / "external"
    link.symlink_to(outside, target_is_directory=True)

    result = service.list_directory(".")

    external_entry = next(
        item for item in result["entries"]
        if item["name"] == "external"
    )

    assert external_entry["kind"] == "symlink"
