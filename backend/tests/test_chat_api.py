
import pytest

from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.api.dependencies import get_provider_manager
from backend.app.intelligence.base import LLMResponse
from backend.app.intelligence.errors import (
    ProviderAuthenticationError,
    ProviderRequestError,
    ProviderTimeout,
    ProviderUnavailable,
)


class FakeProvider:

    def __init__(
        self,
        provider_name="gemini",
        error_to_raise=None,
    ):
        self.provider_name = provider_name
        self.error_to_raise = error_to_raise

    async def generate(self, request):

        if self.error_to_raise is not None:
            raise self.error_to_raise

        assert request.messages[-1].content == "Hello SILVANA"

        return LLMResponse(
            content="Hello. I am SILVANA.",
            provider=self.provider_name,
            model="test-model",
            input_tokens=10,
            output_tokens=8,
            finish_reason="STOP",
        )


class FakeManager:

    def __init__(
        self,
        should_fail=False,
        provider_error=None,
    ):
        self.should_fail = should_fail
        self.provider_error = provider_error

    def available_providers(self):
        return ["gemini", "groq"]

    def get_default_provider_name(self):
        return "gemini"

    def get_provider(self, name):

        if self.should_fail:
            raise ValueError("Provider is not configured")

        return FakeProvider(
            provider_name=name,
            error_to_raise=self.provider_error,
        )


def install_fake_manager(monkeypatch, manager=None):

    if manager is None:
        manager = FakeManager()

    monkeypatch.setitem(
        app.dependency_overrides,
        get_provider_manager,
        lambda: manager,
    )


def test_chat_endpoint(monkeypatch):
    install_fake_manager(monkeypatch)

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/chat",
            json={
                "messages": [
                    {
                        "role": "user",
                        "content": "Hello SILVANA",
                    }
                ]
            },
        )

    assert response.status_code == 200

    data = response.json()

    assert data["content"] == "Hello. I am SILVANA."
    assert data["provider"] == "gemini"
    assert data["model"] == "test-model"
    assert data["input_tokens"] == 10
    assert data["output_tokens"] == 8


def test_chat_endpoint_selects_requested_provider(monkeypatch):
    install_fake_manager(monkeypatch)

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/chat",
            json={
                "messages": [
                    {
                        "role": "user",
                        "content": "Hello SILVANA",
                    }
                ],
                "provider": "groq",
            },
        )

    assert response.status_code == 200
    assert response.json()["provider"] == "groq"


def test_chat_endpoint_rejects_unknown_provider(monkeypatch):
    install_fake_manager(monkeypatch)

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/chat",
            json={
                "messages": [
                    {
                        "role": "user",
                        "content": "Hello SILVANA",
                    }
                ],
                "provider": "unknown",
            },
        )

    assert response.status_code == 422


def test_chat_endpoint_handles_unconfigured_provider(monkeypatch):
    install_fake_manager(
        monkeypatch,
        FakeManager(should_fail=True),
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/chat",
            json={
                "messages": [
                    {
                        "role": "user",
                        "content": "Hello SILVANA",
                    }
                ]
            },
        )

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "The selected AI provider is not configured."
    )


def test_chat_endpoint_rejects_invalid_request(monkeypatch):
    install_fake_manager(monkeypatch)

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/chat",
            json={
                "messages": [],
                "provider": "gemini",
            },
        )

    assert response.status_code == 422


@pytest.mark.parametrize(
    ("provider_error", "expected_status"),
    [
        (ProviderTimeout("Simulated timeout"), 504),
        (ProviderUnavailable("Simulated outage"), 503),
        (
            ProviderAuthenticationError("Simulated authentication failure"),
            502,
        ),
        (ProviderRequestError("Simulated request failure"), 502),
        (RuntimeError("Simulated unexpected failure"), 502),
    ],
)
def test_chat_endpoint_handles_provider_failures(
    monkeypatch,
    provider_error,
    expected_status,
):
    install_fake_manager(
        monkeypatch,
        FakeManager(provider_error=provider_error),
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/chat",
            json={
                "messages": [
                    {
                        "role": "user",
                        "content": "Hello SILVANA",
                    }
                ]
            },
        )

    assert response.status_code == expected_status
    assert "detail" in response.json()
