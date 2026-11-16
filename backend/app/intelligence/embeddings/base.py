from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """Interface for document and query embedding providers."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the embedding provider identifier."""
        raise NotImplementedError

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the embedding model identifier."""
        raise NotImplementedError

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Return the output vector dimensions."""
        raise NotImplementedError

    @abstractmethod
    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """Generate embeddings for multiple document texts."""
        raise NotImplementedError

    @abstractmethod
    def embed_query(
        self,
        text: str,
    ) -> list[float]:
        """Generate an embedding for a single search query."""
        raise NotImplementedError