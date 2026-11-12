
from google import genai
from google.genai import types

from backend.app.core.config import settings
from backend.app.intelligence.base import (
    LLMProvider,
    LLMRequest,
    LLMResponse,
)


class GeminiProvider(LLMProvider):

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

    async def generate(self, request: LLMRequest) -> LLMResponse:

        system_messages = [
            message.content
            for message in request.messages
            if message.role == "system"
        ]

        contents = [
            types.Content(
                role="model" if message.role == "assistant" else "user",
                parts=[types.Part.from_text(text=message.content)]
            )
            for message in request.messages
            if message.role != "system"
        ]

        if not contents:
            raise ValueError("At least one user or assistant message is required.")

        config = types.GenerateContentConfig(
            temperature=request.temperature,
            max_output_tokens=request.max_output_tokens,
            system_instruction="\n\n".join(system_messages) or None,
        )

        response = await self._async_client.models.generate_content(
            model=self._model,
            contents=contents,
            config=config,
        )

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
        await self._async_client.aclose()
        self._client.close()
