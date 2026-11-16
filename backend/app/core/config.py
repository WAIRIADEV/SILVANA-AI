
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    SILVANA_ENV: str = "development"

    GEMINI_API_KEY: str = ""
    GROQ_API_KEY: str = ""

    GEMINI_MODEL: str = "gemini-3.8-flash"
    GROQ_MODEL: str = ""

    DEFAULT_PROVIDER: str = "gemini"

    # Filesystem access is restricted to this directory.
    SILVANA_WORKSPACE_ROOT: Path = (
        Path(__file__).resolve().parents[3] / "workspace"
    )

    # Private document storage, isolated from the user-accessible workspace.
    SILVANA_DOCUMENT_STORAGE_ROOT: Path = (
        Path(__file__).resolve().parents[3] / "storage" / "documents"
    )

    # Maximum size allowed for an individual uploaded document.
    SILVANA_DOCUMENT_MAX_FILE_SIZE_BYTES: int = 20 * 1024 * 1024

    # Temporary conversation documents expire 24 hours after last activity.
    SILVANA_DOCUMENT_TEMPORARY_RETENTION_HOURS: int = 24

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
