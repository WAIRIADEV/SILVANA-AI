
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field, model_validator


class ChatMessageSchema(BaseModel):
    """A single message in a conversation."""

    role: Literal["user", "assistant"]

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

    conversation_id: UUID | None = None

    knowledge_scope: Literal[
        "none",
        "conversation",
        "library",
        "both",
    ] = "none"

    @model_validator(mode="after")
    def validate_knowledge_scope(self):
        """Require ownership context for temporary document access."""

        if (
            self.knowledge_scope in {"conversation", "both"}
            and self.conversation_id is None
        ):
            raise ValueError(
                "A conversation ID is required when knowledge scope "
                "is 'conversation' or 'both'."
            )

        return self


class ChatSourceCitation(BaseModel):
    """A document source used to ground an assistant response."""

    ref: str
    document_id: UUID
    filename: str
    scope: Literal["temporary", "persistent"]
    page_number: int | None = None
    chunk_index: int


class ChatResponse(BaseModel):
    """Structured response returned by SILVANA."""

    content: str
    provider: str
    model: str

    input_tokens: int | None = None
    output_tokens: int | None = None
    finish_reason: str | None = None

    sources: list[ChatSourceCitation] = Field(
        default_factory=list,
    )
