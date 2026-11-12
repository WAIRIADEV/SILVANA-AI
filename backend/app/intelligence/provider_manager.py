
from backend.app.core.config import settings

from backend.app.intelligence.base import LLMProvider
from backend.app.intelligence.providers.gemini import GeminiProvider
from backend.app.intelligence.providers.groq import GroqProvider


class ProviderManager:
    """
    Central manager for SILVANA's LLM providers.

    Handles provider registration, selection, initialization,
    instance reuse, and resource cleanup.
    """

    def __init__(self):
        self._provider_factories = {
            "gemini": GeminiProvider,
            "groq": GroqProvider,
        }

        self._providers: dict[str, LLMProvider] = {}

    def available_providers(self) -> list[str]:
        """Return all registered provider names."""
        return list(self._provider_factories.keys())

    def get_default_provider_name(self) -> str:
        """Return the configured default provider."""
        return settings.DEFAULT_PROVIDER.strip().lower()

    def get_provider(
        self,
        provider_name: str | None = None,
    ) -> LLMProvider:
        """
        Retrieve an initialized provider.

        Uses the configured default when no provider is specified.
        """

        selected_provider = (
            provider_name or self.get_default_provider_name()
        ).strip().lower()

        if selected_provider not in self._provider_factories:
            available = ", ".join(self.available_providers())

            raise ValueError(
                f"Unknown provider '{selected_provider}'. "
                f"Available providers: {available}"
            )

        # Return the existing instance if already initialized.
        if selected_provider in self._providers:
            return self._providers[selected_provider]

        # Initialize the requested provider.
        provider_factory = self._provider_factories[selected_provider]
        provider = provider_factory()

        self._providers[selected_provider] = provider

        return provider

    async def aclose(self) -> None:
        """Close all initialized provider clients."""

        for provider in self._providers.values():
            close_method = getattr(provider, "aclose", None)

            if close_method is not None:
                await close_method()

        self._providers.clear()
