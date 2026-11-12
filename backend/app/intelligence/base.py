from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal


@dataclass
class ChatMessage:
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass
class LLMRequest:
    messages: list[ChatMessage]
    temperature: float = 0.7
    max_output_tokens: int = 1024


@dataclass
class LLMResponse:
    content: str
    provider: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    finish_reason: str | None = None


class LLMProvider(ABC):

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider's identifier."""
        pass

    @abstractmethod
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a response from the supplied conversation."""
        pass
