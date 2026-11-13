
import asyncio
import logging

from backend.app.core.config import settings

from backend.app.intelligence.base import LLMProvider
from backend.app.intelligence.errors import (
    ProviderAuthenticationError,
    ProviderError,
    ProviderRequestError,
    ProviderTimeout,
    ProviderUnavailable,
)
from backend.app.intelligence.providers.gemini import GeminiProvider
from backend.app.intelligence.providers.groq import GroqProvider


logger = logging.getLogger(__name__)


class ProviderManager:
    """
    Central manager for SILVANA's LLM providers.

    Handles provider registration, selection, initialization,
    health monitoring, instance reuse, and resource cleanup.
    """

    HEALTH_CHECK_TIMEOUT = 8.0

    def __init__(self):
        self._provider_factories = {
            "gemini": GeminiProvider,
            "groq": GroqProvider,
        }

        self._providers: dict[str, LLMProvider] = {}

    def available_providers(self) -> list[str]:
        """Return all registered provider names."""
        return list(self._provider_factories.keys())

    def initialized_providers(self) -> list[str]:
        """Return names of currently initialized providers."""
        return list(self._providers.keys())

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

        if selected_provider in self._providers:
            return self._providers[selected_provider]

        provider_factory = self._provider_factories[selected_provider]
        provider = provider_factory()

        self._providers[selected_provider] = provider

        return provider

    def _is_configured(self, provider_name: str) -> bool:
        """Check whether the provider has credentials and a model."""

        configuration = {
            "gemini": (
                settings.GEMINI_API_KEY,
                settings.GEMINI_MODEL,
            ),
            "groq": (
                settings.GROQ_API_KEY,
                settings.GROQ_MODEL,
            ),
        }

        credentials, model = configuration[provider_name]

        return bool(credentials.strip() and model.strip())

    async def check_provider_health(
        self,
        provider_name: str,
    ) -> dict:
        """
        Check authenticated connectivity without unnecessarily
        retaining newly created provider clients.
        """

        selected_provider = provider_name.strip().lower()

        if selected_provider not in self._provider_factories:
            raise ValueError(
                f"Unknown provider '{selected_provider}'."
            )

        initialized = selected_provider in self._providers

        result = {
            "provider": selected_provider,
            "status": "not_configured",
            "configured": False,
            "initialized": initialized,
            "connected": None,
            "message": "Provider credentials or model are not configured.",
        }

        if not self._is_configured(selected_provider):
            return result

        result["configured"] = True

        provider = self._providers.get(selected_provider)
        temporary_client = provider is None

        try:
            if temporary_client:
                provider_factory = self._provider_factories[
                    selected_provider
                ]
                provider = provider_factory()

            await asyncio.wait_for(
                provider.check_connection(),
                timeout=self.HEALTH_CHECK_TIMEOUT,
            )

            result.update({
                "status": "connected",
                "connected": True,
                "message": "Provider connection verified.",
            })

        except asyncio.TimeoutError:
            result.update({
                "status": "timeout",
                "connected": False,
                "message": "Provider health check timed out.",
            })

        except ProviderAuthenticationError:
            result.update({
                "status": "authentication_error",
                "connected": False,
                "message": "Provider authentication failed.",
            })

        except ProviderTimeout:
            result.update({
                "status": "timeout",
                "connected": False,
                "message": "Provider health check timed out.",
            })

        except ProviderUnavailable:
            result.update({
                "status": "unavailable",
                "connected": False,
                "message": "Provider is temporarily unavailable.",
            })

        except ProviderRequestError:
            result.update({
                "status": "error",
                "connected": False,
                "message": "Provider rejected the health check.",
            })

        except ProviderError:
            result.update({
                "status": "error",
                "connected": False,
                "message": "Provider health check failed.",
            })

        except Exception:
            logger.exception(
                "Unexpected health check failure for provider: %s",
                selected_provider,
            )

            result.update({
                "status": "error",
                "connected": False,
                "message": "An unexpected provider health check error occurred.",
            })

        finally:
            if temporary_client and provider is not None:
                close_method = getattr(provider, "aclose", None)

                if close_method is not None:
                    try:
                        await close_method()
                    except Exception:
                        logger.exception(
                            "Failed to close temporary provider client: %s",
                            selected_provider,
                        )

        return result

    async def check_all_provider_health(self) -> dict:
        """Check all registered providers concurrently."""

        results = await asyncio.gather(
            *(
                self.check_provider_health(name)
                for name in self.available_providers()
            )
        )

        return {
            "default_provider": self.get_default_provider_name(),
            "providers": {
                result["provider"]: result
                for result in results
            },
        }

    async def aclose(self) -> None:
        """Close all initialized provider clients."""

        for provider in self._providers.values():
            close_method = getattr(provider, "aclose", None)

            if close_method is not None:
                await close_method()

        self._providers.clear()
