
from groq import AsyncGroq

from backend.app.core.config import settings
from backend.app.intelligence.base import (
    LLMProvider,
    LLMRequest,
    LLMResponse,
)


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

    async def generate(self, request: LLMRequest) -> LLMResponse:

        messages = [
            {
                "role": message.role,
                "content": message.content,
            }
            for message in request.messages
        ]

        completion = await self._client.chat.completions.create(
            messages=messages,
            model=self._model,
            temperature=request.temperature,
            max_completion_tokens=request.max_output_tokens,
        )

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
