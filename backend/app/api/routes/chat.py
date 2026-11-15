
import logging

from fastapi import APIRouter, Depends, HTTPException

from backend.app.api.dependencies import (
    get_provider_manager,
    get_tool_orchestrator,
    get_tool_planner,
)
from backend.app.api.schemas.chat import ChatRequest, ChatResponse
from backend.app.intelligence.base import ChatMessage, LLMRequest
from backend.app.intelligence.conversation_context import (
    ConversationContextAnalyzer,
)
from backend.app.intelligence.identity import SILVANA_SYSTEM_INSTRUCTION
from backend.app.intelligence.intent_engine import Intent, IntentEngine
from backend.app.intelligence.errors import (
    ProviderAuthenticationError,
    ProviderRequestError,
    ProviderTimeout,
    ProviderUnavailable,
)
from backend.app.intelligence.provider_manager import ProviderManager
from backend.app.intelligence.response_engine import ResponseEngine
from backend.app.intelligence.tool_orchestrator import ToolOrchestrator
from backend.app.intelligence.tool_planner import (
    ToolPlanStatus,
    ToolPlanner,
)


logger = logging.getLogger(__name__)

router = APIRouter()

context_analyzer = ConversationContextAnalyzer()
intent_engine = IntentEngine()
response_engine = ResponseEngine()


@router.post("/chat", response_model=ChatResponse)
async def create_chat_completion(
    request: ChatRequest,
    manager: ProviderManager = Depends(get_provider_manager),
    orchestrator: ToolOrchestrator = Depends(get_tool_orchestrator),
    planner: ToolPlanner = Depends(get_tool_planner),
):
    # Convert the incoming conversation into internal messages.
    conversation_messages = [
        ChatMessage(
            role=message.role,
            content=message.content,
        )
        for message in request.messages
    ]

    # Extract context from the conversation history.
    conversation_context = context_analyzer.analyze(
        conversation_messages
    )

    # Identify the latest user request.
    latest_user_message = (
        conversation_context.latest_user_message or ""
    )

    # Classify the request using its conversation context.
    intent_result = intent_engine.classify(
        latest_user_message,
        context=conversation_context,
    )

    # Record classification metadata without logging user content.
    logger.info(
        (
            "Request classified. Intent: %s, Confidence: %.2f, "
            "Follow-up: %s, Correction: %s, New topic: %s, "
            "Context rule: %s"
        ),
        intent_result.intent.value,
        intent_result.confidence,
        intent_result.is_follow_up,
        intent_result.is_correction,
        intent_result.is_new_topic,
        intent_result.context_rule,
    )

    # ---------------------------------------------------------
    # TOOL EXECUTION PATH
    # ---------------------------------------------------------

    if intent_result.intent == Intent.TOOL_ACTION:

        # Discover available capabilities without executing anything.
        available_capabilities = (
            orchestrator.capability_resolver.resolve(intent_result)
        )

        # Create a deterministic plan from the user's request.
        plan = planner.plan(
            intent_result,
            latest_user_message,
            available_capabilities,
        )

        # Ask for clarification when the request is ambiguous.
        if plan.status == ToolPlanStatus.CLARIFICATION_REQUIRED:
            return ChatResponse(
                content=plan.message,
                provider="internal",
                model="tool-planner",
            )

        # Do not silently send unsupported actions to an LLM.
        if plan.status == ToolPlanStatus.UNSUPPORTED:
            return ChatResponse(
                content=plan.message,
                provider="internal",
                model="tool-planner",
            )

        # Only execute plans explicitly marked READY.
        if plan.status != ToolPlanStatus.READY:
            logger.error(
                "Tool planner returned an unexpected status: %s",
                plan.status,
            )

            raise HTTPException(
                status_code=500,
                detail="SILVANA could not safely process the tool plan.",
            )

        # Execute the selected capability through the orchestrator.
        # The orchestrator performs its own capability validation,
        # security checks, execution, and independent verification.
        orchestration_result = await orchestrator.orchestrate(
            intent=intent_result,
            arguments=plan.arguments,
            selected_tool=plan.tool_name,
            user_confirmed=False,
        )

        # Convert the verified orchestration outcome into a response.
        response = response_engine.generate(orchestration_result)

        return ChatResponse(
            content=response.content,
            provider="internal",
            model="filesystem-tools",
        )

    # ---------------------------------------------------------
    # NORMAL CONVERSATION PATH
    # ---------------------------------------------------------

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
        *conversation_messages,
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
