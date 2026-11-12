
from typing import Literal

from pydantic import BaseModel, Field


class ChatMessageSchema(BaseModel):
    """A single message in a conversation."""

    role: Literal["system", "user", "assistant"]

    content: str = Field(
        min_length=1,
        max_length=12000,
    )


class ChatRequest(BaseModel):
    """Incoming chat request from the client."""

    messages: list[ChatMessageSchema] = Field(
        min_length=1,
        max_length=50,
    )

    provider: Literal["gemini", "groq"] | None = None

    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
    )

    max_output_tokens: int = Field(
        default=1024,
        ge=1,
        le=8192,
    )


class ChatResponse(BaseModel):
    """Structured response returned by SILVANA."""

    content: str
    provider: str
    model: str

    input_tokens: int | None = None
    output_tokens: int | None = None
    finish_reason: str | None = None
