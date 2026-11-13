
import asyncio

import pytest

from backend.app.intelligence.errors import (
    ProviderAuthenticationError,
    ProviderError,
    ProviderRequestError,
    ProviderTimeout,
    ProviderUnavailable,
)
from backend.app.intelligence.provider_manager import ProviderManager


def configure_gemini(monkeypatch):
    """Provide test credentials and a model without exposing real keys."""

    monkeypatch.setattr(
        "backend.app.intelligence.provider_manager.settings.GEMINI_API_KEY",
        "test-gemini-key",
    )
    monkeypatch.setattr(
        "backend.app.intelligence.provider_manager.settings.GEMINI_MODEL",
        "test-gemini-model",
    )


class FakeProvider:
    def __init__(self, error=None):
        self.error = error
        self.closed = False
        self.check_count = 0

    async def check_connection(self):
        self.check_count += 1

        if self.error:
            raise self.error

    async def aclose(self):
        self.closed = True


def test_available_providers():
    manager = ProviderManager()

    assert manager.available_providers() == ["gemini", "groq"]


def test_unknown_provider_raises_error():
    manager = ProviderManager()

    with pytest.raises(ValueError, match="Unknown provider"):
        manager.get_provider("invalid")


def test_provider_is_initialized_only_once(monkeypatch):
    manager = ProviderManager()

    class FakeProvider:
        pass

    calls = []

    def fake_factory():
        calls.append(1)
        return FakeProvider()

    manager._provider_factories["gemini"] = fake_factory

    first = manager.get_provider("gemini")
    second = manager.get_provider("gemini")

    assert first is second
    assert len(calls) == 1


def test_default_provider_selection(monkeypatch):
    manager = ProviderManager()

    class FakeProvider:
        pass

    manager._provider_factories["gemini"] = FakeProvider

    monkeypatch.setattr(
        "backend.app.intelligence.provider_manager.settings.DEFAULT_PROVIDER",
        "gemini",
    )

    provider = manager.get_provider()

    assert isinstance(provider, FakeProvider)


def test_provider_manager_closes_initialized_providers():
    manager = ProviderManager()
    fake_provider = FakeProvider()

    manager._providers["gemini"] = fake_provider

    asyncio.run(manager.aclose())

    assert fake_provider.closed is True
    assert manager._providers == {}


def test_health_check_returns_not_configured_when_credentials_missing(
    monkeypatch,
):
    manager = ProviderManager()

    monkeypatch.setattr(
        "backend.app.intelligence.provider_manager.settings.GEMINI_API_KEY",
        "",
    )

    calls = []

    def fake_factory():
        calls.append(1)
        return FakeProvider()

    manager._provider_factories["gemini"] = fake_factory

    result = asyncio.run(
        manager.check_provider_health("gemini")
    )

    assert result["status"] == "not_configured"
    assert result["configured"] is False
    assert result["connected"] is None
    assert calls == []


def test_health_check_success_closes_temporary_provider(monkeypatch):
    configure_gemini(monkeypatch)

    manager = ProviderManager()
    fake_provider = FakeProvider()

    manager._provider_factories["gemini"] = lambda: fake_provider

    result = asyncio.run(
        manager.check_provider_health("gemini")
    )

    assert result["status"] == "connected"
    assert result["configured"] is True
    assert result["initialized"] is False
    assert result["connected"] is True

    assert fake_provider.check_count == 1
    assert fake_provider.closed is True

    # Temporary health-check clients must not be retained.
    assert manager._providers == {}


def test_health_check_reuses_initialized_provider(monkeypatch):
    configure_gemini(monkeypatch)

    manager = ProviderManager()
    fake_provider = FakeProvider()

    manager._providers["gemini"] = fake_provider

    result = asyncio.run(
        manager.check_provider_health("gemini")
    )

    assert result["status"] == "connected"
    assert result["initialized"] is True
    assert fake_provider.check_count == 1

    # An existing client belongs to the manager and must remain open.
    assert fake_provider.closed is False
    assert manager._providers["gemini"] is fake_provider


@pytest.mark.parametrize(
    ("error", "expected_status"),
    [
        (
            ProviderAuthenticationError("Invalid credentials"),
            "authentication_error",
        ),
        (
            ProviderTimeout("Request timed out"),
            "timeout",
        ),
        (
            ProviderUnavailable("Provider unavailable"),
            "unavailable",
        ),
        (
            ProviderRequestError("Request rejected"),
            "error",
        ),
        (
            ProviderError("Generic provider failure"),
            "error",
        ),
        (
            RuntimeError("Unexpected failure"),
            "error",
        ),
    ],
)
def test_health_check_maps_provider_errors(
    monkeypatch,
    error,
    expected_status,
):
    configure_gemini(monkeypatch)

    manager = ProviderManager()
    fake_provider = FakeProvider(error=error)

    manager._provider_factories["gemini"] = lambda: fake_provider

    result = asyncio.run(
        manager.check_provider_health("gemini")
    )

    assert result["status"] == expected_status
    assert result["connected"] is False
    assert fake_provider.closed is True
    assert manager._providers == {}


def test_health_check_enforces_timeout_and_closes_client(monkeypatch):
    configure_gemini(monkeypatch)

    manager = ProviderManager()
    manager.HEALTH_CHECK_TIMEOUT = 0.01

    class SlowProvider(FakeProvider):
        async def check_connection(self):
            await asyncio.sleep(0.1)

    fake_provider = SlowProvider()

    manager._provider_factories["gemini"] = lambda: fake_provider

    result = asyncio.run(
        manager.check_provider_health("gemini")
    )

    assert result["status"] == "timeout"
    assert result["connected"] is False
    assert fake_provider.closed is True
    assert manager._providers == {}


def test_health_check_rejects_unknown_provider():
    manager = ProviderManager()

    with pytest.raises(ValueError, match="Unknown provider"):
        asyncio.run(
            manager.check_provider_health("invalid")
        )
