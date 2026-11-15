
import pytest

from backend.app.intelligence.filesystem.workspace import (
    WorkspaceBoundary,
    WorkspacePathError,
)


@pytest.fixture
def workspace(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()

    (root / "Documents").mkdir()
    (root / "Documents" / "report.txt").write_text(
        "SILVANA workspace test",
        encoding="utf-8",
    )

    return WorkspaceBoundary(root), root


def test_resolves_workspace_relative_path(workspace):
    boundary, root = workspace

    result = boundary.resolve("Documents/report.txt")

    assert result == root / "Documents" / "report.txt"


def test_resolves_workspace_root(workspace):
    boundary, root = workspace

    assert boundary.resolve(".") == root


@pytest.mark.parametrize(
    "path",
    [
        "../outside.txt",
        "../../secret.txt",
        "Documents/../../secret.txt",
        "..\\outside.txt",
        "C:\\Windows\\win.ini",
        "\\\\server\\share\\secret.txt",
        "/etc/passwd",
    ],
)
def test_rejects_absolute_and_traversal_paths(workspace, path):
    boundary, _ = workspace

    with pytest.raises(WorkspacePathError):
        boundary.resolve(path, must_exist=False)


def test_rejects_null_bytes(workspace):
    boundary, _ = workspace

    with pytest.raises(WorkspacePathError):
        boundary.resolve("Documents/\x00report.txt")


def test_rejects_empty_path(workspace):
    boundary, _ = workspace

    with pytest.raises(WorkspacePathError):
        boundary.resolve("")


def test_rejects_non_string_path(workspace):
    boundary, _ = workspace

    with pytest.raises(WorkspacePathError):
        boundary.resolve(None)


def test_rejects_missing_path_when_required(workspace):
    boundary, _ = workspace

    with pytest.raises(WorkspacePathError):
        boundary.resolve("missing.txt")


def test_allows_missing_final_path_for_existence_checks(workspace):
    boundary, root = workspace

    result = boundary.resolve(
        "Documents/missing.txt",
        must_exist=False,
    )

    assert result == root / "Documents" / "missing.txt"


def test_rejects_symlink_escape(workspace, tmp_path):
    boundary, _ = workspace

    outside = tmp_path / "outside"
    outside.mkdir()

    link = boundary.root / "external"
    link.symlink_to(outside, target_is_directory=True)

    with pytest.raises(WorkspacePathError):
        boundary.resolve("external")


def test_rejects_non_directory_workspace(tmp_path):
    file_path = tmp_path / "not_a_directory.txt"
    file_path.write_text("test", encoding="utf-8")

    with pytest.raises(WorkspacePathError):
        WorkspaceBoundary(file_path)
