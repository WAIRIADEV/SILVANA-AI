
import logging

from google import genai
from google.genai import types
from google.genai.errors import APIError

from backend.app.core.config import settings
from backend.app.intelligence.base import (
    LLMProvider,
    LLMRequest,
    LLMResponse,
)
from backend.app.intelligence.errors import (
    ProviderError,
    normalize_status_error,
)


logger = logging.getLogger(__name__)


class GeminiProvider(LLMProvider):
    """Gemini implementation of SILVANA's LLM provider interface."""

    def __init__(self):
        if not settings.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is not configured.")

        if not settings.GEMINI_MODEL:
            raise ValueError("GEMINI_MODEL is not configured.")

        self._client = genai.Client(
            api_key=settings.GEMINI_API_KEY
        )
        self._async_client = self._client.aio
        self._model = settings.GEMINI_MODEL

    @property
    def provider_name(self) -> str:
        return "gemini"

    def _normalize_error(self, error: APIError) -> ProviderError:
        """Translate Gemini API errors into SILVANA exceptions."""

        status_code = getattr(error, "code", None)

        if not isinstance(status_code, int):
            status_code = getattr(error, "status", None)

        if not isinstance(status_code, int):
            return ProviderError(
                "The Gemini provider encountered an unexpected error."
            )

        return normalize_status_error(
            status_code=status_code,
            message=str(getattr(error, "message", "")),
        )

    async def check_connection(self) -> None:
        """Verify authenticated connectivity to the Gemini API."""

        try:
            await self._async_client.models.list()

        except APIError as error:
            logger.warning(
                "Gemini health check failed. Status: %s",
                getattr(error, "code", None),
            )

            raise self._normalize_error(error) from error

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a response using the configured Gemini model."""

        system_messages = [
            message.content
            for message in request.messages
            if message.role == "system"
        ]

        contents = [
            types.Content(
                role="model" if message.role == "assistant" else "user",
                parts=[types.Part.from_text(text=message.content)],
            )
            for message in request.messages
            if message.role != "system"
        ]

        if not contents:
            raise ValueError(
                "At least one user or assistant message is required."
            )

        config = types.GenerateContentConfig(
            temperature=request.temperature,
            max_output_tokens=request.max_output_tokens,
            system_instruction="\n\n".join(system_messages) or None,
        )

        try:
            response = await self._async_client.models.generate_content(
                model=self._model,
                contents=contents,
                config=config,
            )

        except APIError as error:
            logger.warning(
                "Gemini API request failed. Status: %s",
                getattr(error, "code", None),
            )

            raise self._normalize_error(error) from error

        usage = response.usage_metadata

        finish_reason = None

        if response.candidates:
            reason = response.candidates[0].finish_reason
            finish_reason = str(reason) if reason else None

        return LLMResponse(
            content=response.text or "",
            provider=self.provider_name,
            model=self._model,
            input_tokens=(
                usage.prompt_token_count if usage else None
            ),
            output_tokens=(
                usage.candidates_token_count if usage else None
            ),
            finish_reason=finish_reason,
        )

    async def aclose(self):
        """Close Gemini SDK clients and release resources."""

        await self._async_client.aclose()
        self._client.close()
