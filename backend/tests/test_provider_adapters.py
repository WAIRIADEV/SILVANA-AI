
import asyncio
from types import SimpleNamespace

import httpx
import pytest

from google.genai.errors import APIError
from groq import APIStatusError, APITimeoutError, APIConnectionError

from backend.app.intelligence.base import ChatMessage, LLMRequest
from backend.app.intelligence.errors import (
    ProviderAuthenticationError,
    ProviderRequestError,
    ProviderTimeout,
    ProviderUnavailable,
)
from backend.app.intelligence.providers.gemini import GeminiProvider
from backend.app.intelligence.providers.groq import GroqProvider


def make_request():
    return LLMRequest(
        messages=[
            ChatMessage(
                role="user",
                content="Hello SILVANA",
            )
        ]
    )


def make_gemini_provider(error):
    provider = GeminiProvider.__new__(GeminiProvider)

    async def simulate_failure(**kwargs):
        raise error

    provider._async_client = SimpleNamespace(
        models=SimpleNamespace(
            generate_content=simulate_failure,
        )
    )
    provider._model = "test-model"

    return provider


def make_groq_provider(error):
    provider = GroqProvider.__new__(GroqProvider)

    async def simulate_failure(**kwargs):
        raise error

    provider._client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=simulate_failure,
            )
        )
    )
    provider._model = "test-model"

    return provider


@pytest.mark.parametrize(
    ("status_code", "expected_error"),
    [
        (503, ProviderUnavailable),
        (429, ProviderUnavailable),
        (401, ProviderAuthenticationError),
        (504, ProviderTimeout),
        (400, ProviderRequestError),
    ],
)
def test_gemini_normalizes_api_errors(
    status_code,
    expected_error,
):
    error = APIError(
        status_code,
        {
            "error": {
                "message": "Simulated Gemini failure",
                "status": "SIMULATED",
            }
        },
    )

    provider = make_gemini_provider(error)

    with pytest.raises(expected_error):
        asyncio.run(provider.generate(make_request()))


@pytest.mark.parametrize(
    ("status_code", "expected_error"),
    [
        (503, ProviderUnavailable),
        (429, ProviderUnavailable),
        (401, ProviderAuthenticationError),
        (504, ProviderTimeout),
        (400, ProviderRequestError),
    ],
)
def test_groq_normalizes_api_status_errors(
    status_code,
    expected_error,
):
    request = httpx.Request(
        "POST",
        "https://api.groq.com/openai/v1/chat/completions",
    )

    response = httpx.Response(
        status_code,
        request=request,
    )

    error = APIStatusError(
        "Simulated Groq failure",
        response=response,
        body=None,
    )

    provider = make_groq_provider(error)

    with pytest.raises(expected_error):
        asyncio.run(provider.generate(make_request()))


def test_groq_normalizes_timeout_error():
    request = httpx.Request(
        "POST",
        "https://api.groq.com/openai/v1/chat/completions",
    )

    error = APITimeoutError(request)

    provider = make_groq_provider(error)

    with pytest.raises(ProviderTimeout):
        asyncio.run(provider.generate(make_request()))


def test_groq_normalizes_connection_error():
    request = httpx.Request(
        "POST",
        "https://api.groq.com/openai/v1/chat/completions",
    )

    error = APIConnectionError(
        message="Simulated connection failure",
        request=request,
    )

    provider = make_groq_provider(error)

    with pytest.raises(ProviderUnavailable):
        asyncio.run(provider.generate(make_request()))
