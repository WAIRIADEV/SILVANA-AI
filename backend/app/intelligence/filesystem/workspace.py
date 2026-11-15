
from pathlib import Path, PureWindowsPath


class WorkspacePathError(ValueError):
    """Raised when a requested path violates workspace restrictions."""


class WorkspaceBoundary:
    """
    Restricts filesystem access to one explicitly permitted directory.

    All paths supplied by callers are interpreted relative to the
    workspace root. Absolute paths and traversal attempts are rejected.
    """

    def __init__(self, root: str | Path):
        try:
            resolved_root = Path(root).expanduser().resolve(strict=True)
        except (OSError, RuntimeError) as error:
            raise WorkspacePathError(
                "The configured workspace root cannot be accessed."
            ) from error

        if not resolved_root.is_dir():
            raise WorkspacePathError(
                "The configured workspace root must be a directory."
            )

        self.root = resolved_root

    def resolve(
        self,
        requested_path: str,
        *,
        must_exist: bool = True,
    ) -> Path:
        """
        Resolve a workspace-relative path without permitting escape.

        must_exist=False is intended for existence checks where the
        final requested path may not exist yet.
        """

        if not isinstance(requested_path, str):
            raise WorkspacePathError(
                "The requested path must be a string."
            )

        if not requested_path or not requested_path.strip():
            raise WorkspacePathError(
                "The requested path cannot be empty."
            )

        if "\x00" in requested_path:
            raise WorkspacePathError(
                "Null bytes are not permitted in paths."
            )

        # Reject POSIX and Windows absolute paths, including drive paths.
        windows_path = PureWindowsPath(requested_path)

        if (
            Path(requested_path).is_absolute()
            or windows_path.is_absolute()
            or windows_path.drive
        ):
            raise WorkspacePathError(
                "Absolute paths are not permitted."
            )

        # Reject traversal regardless of slash direction.
        components = requested_path.replace("\\", "/").split("/")

        if ".." in components:
            raise WorkspacePathError(
                "Directory traversal is not permitted."
            )

        # Reject Windows alternate data stream syntax.
        if ":" in requested_path:
            raise WorkspacePathError(
                "Invalid path syntax."
            )

        candidate = self.root / requested_path

        try:
            resolved_path = candidate.resolve(strict=must_exist)
        except FileNotFoundError as error:
            raise WorkspacePathError(
                "The requested path does not exist."
            ) from error
        except (OSError, RuntimeError) as error:
            raise WorkspacePathError(
                "The requested path cannot be accessed."
            ) from error

        # Resolve symlinks before checking containment.
        try:
            resolved_path.relative_to(self.root)
        except ValueError as error:
            raise WorkspacePathError(
                "The requested path is outside the permitted workspace."
            ) from error

        return resolved_path
