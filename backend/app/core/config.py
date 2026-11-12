from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    SILVANA_ENV: str = "development"

    GEMINI_API_KEY: str = ""
    GROQ_API_KEY: str = ""

    GEMINI_MODEL: str = "gemini-3.8-flash"
    GROQ_MODEL: str = ""

    DEFAULT_PROVIDER: str = "gemini"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()
