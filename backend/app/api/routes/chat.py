
import logging

from fastapi import APIRouter, Depends, HTTPException

from backend.app.api.dependencies import get_provider_manager
from backend.app.api.schemas.chat import ChatRequest, ChatResponse
from backend.app.intelligence.base import ChatMessage, LLMRequest
from backend.app.intelligence.identity import SILVANA_SYSTEM_INSTRUCTION
from backend.app.intelligence.errors import (
    ProviderAuthenticationError,
    ProviderRequestError,
    ProviderTimeout,
    ProviderUnavailable,
)
from backend.app.intelligence.provider_manager import ProviderManager


logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def create_chat_completion(
    request: ChatRequest,
    manager: ProviderManager = Depends(get_provider_manager),
):
    # Resolve the requested or configured default provider.
    selected_provider = (
        request.provider or manager.get_default_provider_name()
    ).strip().lower()

    # Reject unknown providers.
    if selected_provider not in manager.available_providers():
        raise HTTPException(
            status_code=400,
            detail="The requested AI provider is not supported.",
        )

    # Retrieve or initialize the selected provider.
    try:
        provider = manager.get_provider(selected_provider)

    except ValueError:
        logger.exception(
            "Provider initialization failed: %s",
            selected_provider,
        )

        raise HTTPException(
            status_code=503,
            detail="The selected AI provider is not configured.",
        )

    # Inject SILVANA's trusted identity before conversation history.
    messages = [
        ChatMessage(
            role="system",
            content=SILVANA_SYSTEM_INSTRUCTION,
        ),
        *[
            ChatMessage(
                role=message.role,
                content=message.content,
            )
            for message in request.messages
        ],
    ]

    llm_request = LLMRequest(
        messages=messages,
        temperature=request.temperature,
        max_output_tokens=request.max_output_tokens,
    )

    # Generate a response through the selected provider.
    try:
        result = await provider.generate(llm_request)

    except ProviderTimeout:
        logger.warning(
            "AI provider timed out. Provider: %s",
            selected_provider,
        )

        raise HTTPException(
            status_code=504,
            detail="The AI provider took too long to respond.",
        )

    except ProviderUnavailable:
        logger.warning(
            "AI provider is temporarily unavailable. Provider: %s",
            selected_provider,
        )

        raise HTTPException(
            status_code=503,
            detail="The AI provider is temporarily unavailable.",
        )

    except ProviderAuthenticationError:
        logger.error(
            "AI provider authentication failed. Provider: %s",
            selected_provider,
        )

        raise HTTPException(
            status_code=502,
            detail="The AI provider could not authenticate the request.",
        )

    except ProviderRequestError:
        logger.warning(
            "AI provider rejected the request. Provider: %s",
            selected_provider,
        )

        raise HTTPException(
            status_code=502,
            detail="The AI provider could not process the request.",
        )

    except Exception:
        logger.exception(
            "Unexpected AI provider failure. Provider: %s",
            selected_provider,
        )

        raise HTTPException(
            status_code=502,
            detail="The AI provider failed to complete the request.",
        )

    # Return a consistent structured response.
    return ChatResponse(
        content=result.content,
        provider=result.provider,
        model=result.model,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        finish_reason=result.finish_reason,
    )
