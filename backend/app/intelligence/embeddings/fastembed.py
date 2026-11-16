
from math import isfinite
from typing import Any

from fastembed import TextEmbedding
from tokenizers import Tokenizer

from backend.app.intelligence.embeddings.base import EmbeddingProvider


class FastEmbedProvider(EmbeddingProvider):
    """Local embedding provider powered by FastEmbed."""

    DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
    DEFAULT_BATCH_SIZE = 32
    DEFAULT_THREADS = 2

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        batch_size: int = DEFAULT_BATCH_SIZE,
        threads: int = DEFAULT_THREADS,
        model: Any | None = None,
        tokenizer: Any | None = None,
    ) -> None:
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("Model name must be a non-empty string.")

        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise ValueError("Batch size must be an integer.")

        if batch_size < 1:
            raise ValueError("Batch size must be at least 1.")

        if isinstance(threads, bool) or not isinstance(threads, int):
            raise ValueError("Thread count must be an integer.")

        if threads < 1:
            raise ValueError("Thread count must be at least 1.")

        self._model_name = model_name
        self.batch_size = batch_size
        self.threads = threads
        self._dimensions = self._resolve_dimensions(model_name)
        self._model = model
        self._tokenizer = tokenizer

    @staticmethod
    def _resolve_dimensions(model_name: str) -> int:
        supported_models = TextEmbedding.list_supported_models()

        for model_info in supported_models:
            if model_info.get("model") == model_name:
                dimensions = model_info.get("dim")

                if (
                    isinstance(dimensions, bool)
                    or not isinstance(dimensions, int)
                    or dimensions < 1
                ):
                    raise ValueError(
                        f"Invalid embedding dimensions for {model_name}."
                    )

                return dimensions

        raise ValueError(f"Unsupported FastEmbed model: {model_name}.")

    @property
    def provider_name(self) -> str:
        return "fastembed"

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def _get_model(self) -> Any:
        """Load and cache the embedding model on first use."""

        if self._model is None:
            try:
                self._model = TextEmbedding(
                    model_name=self._model_name,
                    threads=self.threads,
                )
            except Exception as exc:
                raise RuntimeError(
                    "FastEmbed model initialization failed."
                ) from exc

        return self._model

    def _get_tokenizer(self) -> Any:
        """Load an independent tokenizer without truncation."""

        if self._tokenizer is None:
            try:
                model = self._get_model()

                tokenizer = Tokenizer.from_str(
                    model.model.tokenizer.to_str()
                )

                # Only the counting tokenizer has truncation disabled.
                # FastEmbed's inference tokenizer remains unchanged.
                tokenizer.no_truncation()

                self._tokenizer = tokenizer

            except Exception as exc:
                raise RuntimeError(
                    "FastEmbed tokenizer initialization failed."
                ) from exc

        return self._tokenizer

    @staticmethod
    def _validate_text(text: str, field_name: str) -> None:
        if not isinstance(text, str):
            raise TypeError(f"{field_name} must be a string.")

        if not text.strip():
            raise ValueError(f"{field_name} must not be empty.")

    def count_tokens(self, text: str) -> int:
        """Count tokens using the embedding model's exact tokenizer."""

        self._validate_text(text, "Text")
        tokenizer = self._get_tokenizer()

        try:
            encoding = tokenizer.encode(
                text,
                add_special_tokens=True,
            )
            return len(encoding.ids)

        except Exception as exc:
            raise RuntimeError(
                "FastEmbed token counting failed."
            ) from exc

    def _normalize_vector(self, vector: Any) -> list[float]:
        if hasattr(vector, "tolist"):
            vector = vector.tolist()

        if not isinstance(vector, (list, tuple)):
            raise TypeError("Embedding output must be a sequence.")

        if len(vector) != self.dimensions:
            raise ValueError(
                f"Expected {self.dimensions} dimensions, "
                f"received {len(vector)}."
            )

        normalized: list[float] = []

        for value in vector:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError("Embedding values must be numeric.")

            numeric_value = float(value)

            if not isfinite(numeric_value):
                raise ValueError("Embedding values must be finite.")

            normalized.append(numeric_value)

        return normalized

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not isinstance(texts, list):
            raise TypeError("Texts must be provided as a list.")

        if not texts:
            return []

        for index, text in enumerate(texts):
            self._validate_text(text, f"Document at index {index}")

        model = self._get_model()

        try:
            vectors = list(
                model.embed(texts, batch_size=self.batch_size)
            )
        except Exception as exc:
            raise RuntimeError(
                "FastEmbed document embedding failed."
            ) from exc

        if len(vectors) != len(texts):
            raise RuntimeError(
                "FastEmbed returned an unexpected number of vectors."
            )

        return [
            self._normalize_vector(vector)
            for vector in vectors
        ]

    def embed_query(self, text: str) -> list[float]:
        self._validate_text(text, "Query")
        model = self._get_model()

        try:
            vectors = list(model.query_embed(text))
        except Exception as exc:
            raise RuntimeError(
                "FastEmbed query embedding failed."
            ) from exc

        if len(vectors) != 1:
            raise RuntimeError(
                "FastEmbed returned an unexpected number of query vectors."
            )

        return self._normalize_vector(vectors[0])
