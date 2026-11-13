
import logging

from groq import (
    AsyncGroq,
    APIStatusError,
    APITimeoutError,
    APIConnectionError,
)

from backend.app.core.config import settings
from backend.app.intelligence.base import (
    LLMProvider,
    LLMRequest,
    LLMResponse,
)
from backend.app.intelligence.errors import (
    ProviderError,
    ProviderUnavailable,
    ProviderTimeout,
    normalize_status_error,
)


logger = logging.getLogger(__name__)


class GroqProvider(LLMProvider):

    def __init__(self):
        if not settings.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is not configured.")

        if not settings.GROQ_MODEL:
            raise ValueError("GROQ_MODEL is not configured.")

        self._client = AsyncGroq(
            api_key=settings.GROQ_API_KEY
        )
        self._model = settings.GROQ_MODEL

    @property
    def provider_name(self) -> str:
        return "groq"

    async def check_connection(self) -> None:
        """Verify authenticated connectivity to the Groq API."""

        try:
            await self._client.models.list()

        except APITimeoutError as error:
            logger.warning("Groq health check timed out.")

            raise ProviderTimeout(
                "The Groq health check timed out."
            ) from error

        except APIConnectionError as error:
            logger.warning("Groq health check connection failed.")

            raise ProviderUnavailable(
                "The Groq provider is temporarily unreachable."
            ) from error

        except APIStatusError as error:
            logger.warning(
                "Groq health check failed. Status: %s",
                error.status_code,
            )

            raise normalize_status_error(
                status_code=error.status_code,
            ) from error

    async def generate(self, request: LLMRequest) -> LLMResponse:

        messages = [
            {
                "role": message.role,
                "content": message.content,
            }
            for message in request.messages
        ]

        try:
            completion = await self._client.chat.completions.create(
                messages=messages,
                model=self._model,
                temperature=request.temperature,
                max_completion_tokens=request.max_output_tokens,
            )

        except APITimeoutError as error:
            logger.warning("Groq request timed out.")

            raise ProviderTimeout(
                "The Groq provider request timed out."
            ) from error

        except APIConnectionError as error:
            logger.warning("Groq connection failed.")

            raise ProviderUnavailable(
                "The Groq provider is temporarily unreachable."
            ) from error

        except APIStatusError as error:
            logger.warning(
                "Groq API request failed. Status: %s",
                error.status_code,
            )

            raise normalize_status_error(
                status_code=error.status_code,
            ) from error

        choice = completion.choices[0]
        usage = completion.usage

        return LLMResponse(
            content=choice.message.content or "",
            provider=self.provider_name,
            model=self._model,
            input_tokens=(
                usage.prompt_tokens if usage else None
            ),
            output_tokens=(
                usage.completion_tokens if usage else None
            ),
            finish_reason=choice.finish_reason,
        )

    async def aclose(self):
        await self._client.close()
