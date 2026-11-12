
from backend.app.core.config import settings

from backend.app.intelligence.base import LLMProvider
from backend.app.intelligence.providers.gemini import GeminiProvider
from backend.app.intelligence.providers.groq import GroqProvider


class ProviderManager:
    """
    Manages LLM providers and their lifecycle.

    Providers are initialized only when requested
    and reused throughout the application lifecycle.
    """

    def __init__(self):
        self._provider_factories = {
            "gemini": GeminiProvider,
            "groq": GroqProvider,
        }

        self._providers: dict[str, LLMProvider] = {}

    def get_provider(
        self,
        provider_name: str | None = None
    ) -> LLMProvider:
        """
        Return an initialized provider.

        Uses DEFAULT_PROVIDER when no provider is specified.
        """

        selected_provider = (
            provider_name or settings.DEFAULT_PROVIDER
        ).strip().lower()

        if selected_provider not in self._provider_factories:
            available = ", ".join(self._provider_factories)

            raise ValueError(
                f"Unknown provider '{selected_provider}'. "
                f"Available providers: {available}"
            )

        # Reuse an existing provider instance.
        if selected_provider in self._providers:
            return self._providers[selected_provider]

        # Initialize the requested provider.
        provider_factory = self._provider_factories[
            selected_provider
        ]

        provider = provider_factory()

        self._providers[selected_provider] = provider

        return provider

    def available_providers(self) -> list[str]:
        """Return registered provider names."""

        return list(self._provider_factories.keys())

    async def aclose(self):
        """Close all initialized provider clients."""

        for provider in self._providers.values():
            close_method = getattr(provider, "aclose", None)

            if close_method is not None:
                await close_method()

        self._providers.clear()
