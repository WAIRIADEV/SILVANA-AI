
import pytest

from backend.app.intelligence.provider_manager import ProviderManager


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
        "gemini"
    )

    provider = manager.get_provider()

    assert isinstance(provider, FakeProvider)

import asyncio


def test_provider_manager_closes_initialized_providers():
    manager = ProviderManager()

    class FakeProvider:
        def __init__(self):
            self.closed = False

        async def aclose(self):
            self.closed = True

    fake_provider = FakeProvider()

    manager._providers["gemini"] = fake_provider

    asyncio.run(manager.aclose())

    assert fake_provider.closed is True
    assert manager._providers == {}
