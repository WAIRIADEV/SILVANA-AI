
import asyncio

import pytest

from backend.app.intelligence.capability_resolver import CapabilityResolver
from backend.app.intelligence.execution_engine import ExecutionEngine
from backend.app.intelligence.intent_engine import Intent, IntentResult
from backend.app.intelligence.tool_orchestrator import (
    OrchestrationStatus,
    ToolOrchestrator,
)
from backend.app.intelligence.tool_registry import ToolDefinition, ToolRegistry
from backend.app.intelligence.tool_security import ToolSecurityManager
from backend.app.intelligence.verification_engine import (
    VerificationEngine,
    VerificationStatus,
)


@pytest.fixture
def setup_orchestrator():
    registry = ToolRegistry()

    security = ToolSecurityManager(
        granted_permissions={"read", "write"}
    )

    resolver = CapabilityResolver(registry)

    execution_engine = ExecutionEngine(
        registry=registry,
        security_manager=security,
    )

    verification_engine = VerificationEngine()

    orchestrator = ToolOrchestrator(
        capability_resolver=resolver,
        execution_engine=execution_engine,
        verification_engine=verification_engine,
    )

    return registry, orchestrator, verification_engine


def make_intent():
    return IntentResult(
        intent=Intent.TOOL_ACTION,
        confidence=0.95,
        requires_tool=True,
        requires_confirmation=False,
        matched_rule="explicit_action",
    )


def run_orchestration(orchestrator, **kwargs):
    return asyncio.run(
        orchestrator.orchestrate(**kwargs)
    )


def test_no_capability_returns_safe_result(setup_orchestrator):
    _, orchestrator, _ = setup_orchestrator

    result = run_orchestration(
        orchestrator,
        intent=make_intent(),
    )

    assert result.status == OrchestrationStatus.NO_CAPABILITY
    assert result.available_tools == []
    assert result.execution is None
    assert result.verification is None


def test_matching_capability_requires_selection(setup_orchestrator):
    registry, orchestrator, _ = setup_orchestrator

    registry.register(
        ToolDefinition(
            name="send_message",
            description="Sends a message.",
            handler=lambda: "sent",
            supported_intents={"tool_action"},
        )
    )

    result = run_orchestration(
        orchestrator,
        intent=make_intent(),
    )

    assert result.status == OrchestrationStatus.SELECTION_REQUIRED
    assert result.available_tools == ["send_message"]
    assert result.execution is None
    assert result.verification is None


def test_invalid_selection_is_rejected(setup_orchestrator):
    registry, orchestrator, _ = setup_orchestrator

    registry.register(
        ToolDefinition(
            name="send_message",
            description="Sends a message.",
            handler=lambda: "sent",
            supported_intents={"tool_action"},
        )
    )

    result = run_orchestration(
        orchestrator,
        intent=make_intent(),
        selected_tool="delete_file",
    )

    assert result.status == OrchestrationStatus.INVALID_SELECTION
    assert result.execution is None
    assert result.verification is None


def test_execution_and_verification_succeed(setup_orchestrator):
    registry, orchestrator, verification_engine = setup_orchestrator

    state = {"message_sent": False}

    def send_message():
        state["message_sent"] = True
        return "Message handler completed."

    registry.register(
        ToolDefinition(
            name="send_message",
            description="Sends a message.",
            handler=send_message,
            permission_level="write",
            supported_intents={"tool_action"},
        )
    )

    verification_engine.register(
        "send_message",
        lambda result: state["message_sent"] is True,
    )

    result = run_orchestration(
        orchestrator,
        intent=make_intent(),
        selected_tool="send_message",
    )

    assert result.status == OrchestrationStatus.VERIFICATION_SUCCEEDED
    assert result.execution is not None
    assert result.execution.success is True
    assert result.execution.result == "Message handler completed."

    assert result.verification is not None
    assert result.verification.status == VerificationStatus.VERIFIED
    assert result.verification.verified is True


def test_successful_execution_without_verifier_is_unverifiable(
    setup_orchestrator,
):
    registry, orchestrator, _ = setup_orchestrator

    registry.register(
        ToolDefinition(
            name="send_message",
            description="Sends a message.",
            handler=lambda: "Message handler completed.",
            permission_level="write",
            supported_intents={"tool_action"},
        )
    )

    result = run_orchestration(
        orchestrator,
        intent=make_intent(),
        selected_tool="send_message",
    )

    assert result.status == OrchestrationStatus.VERIFICATION_UNVERIFIABLE
    assert result.execution is not None
    assert result.execution.success is True

    assert result.verification is not None
    assert result.verification.status == VerificationStatus.UNVERIFIABLE
    assert result.verification.verified is False


def test_verification_failure_is_not_reported_as_success(
    setup_orchestrator,
):
    registry, orchestrator, verification_engine = setup_orchestrator

    state = {"file_exists": False}

    registry.register(
        ToolDefinition(
            name="create_file",
            description="Creates a file.",
            handler=lambda: "Creation handler completed.",
            permission_level="write",
            supported_intents={"tool_action"},
        )
    )

    verification_engine.register(
        "create_file",
        lambda result: state["file_exists"],
    )

    result = run_orchestration(
        orchestrator,
        intent=make_intent(),
        selected_tool="create_file",
    )

    assert result.status == OrchestrationStatus.VERIFICATION_FAILED
    assert result.execution is not None
    assert result.execution.success is True

    assert result.verification is not None
    assert result.verification.status == VerificationStatus.FAILED
    assert result.verification.verified is False


def test_confirmation_is_enforced(setup_orchestrator):
    registry, orchestrator, _ = setup_orchestrator

    registry.register(
        ToolDefinition(
            name="send_message",
            description="Sends a message.",
            handler=lambda: "sent",
            permission_level="write",
            requires_confirmation=True,
            supported_intents={"tool_action"},
        )
    )

    result = run_orchestration(
        orchestrator,
        intent=make_intent(),
        selected_tool="send_message",
    )

    assert result.status == OrchestrationStatus.EXECUTION_FAILED
    assert result.execution is not None
    assert result.execution.success is False
    assert "confirmation" in result.execution.error.lower()

    assert result.verification is not None
    assert result.verification.status == VerificationStatus.FAILED
    assert result.verification.verified is False


def test_invalid_arguments_prevent_execution(setup_orchestrator):
    registry, orchestrator, _ = setup_orchestrator

    tracker = {"called": False}

    def protected_tool(value):
        tracker["called"] = True
        return value

    registry.register(
        ToolDefinition(
            name="protected_tool",
            description="Accepts an integer.",
            handler=protected_tool,
            parameters={
                "type": "object",
                "properties": {
                    "value": {"type": "integer"},
                },
                "required": ["value"],
                "additionalProperties": False,
            },
            supported_intents={"tool_action"},
        )
    )

    result = run_orchestration(
        orchestrator,
        intent=make_intent(),
        selected_tool="protected_tool",
        arguments={"value": "invalid"},
    )

    assert result.status == OrchestrationStatus.EXECUTION_FAILED
    assert result.execution is not None
    assert result.execution.success is False
    assert tracker["called"] is False

    assert result.verification is not None
    assert result.verification.status == VerificationStatus.FAILED


def test_discovery_never_executes_handler(setup_orchestrator):
    registry, orchestrator, _ = setup_orchestrator

    tracker = {"called": False}

    def protected_tool():
        tracker["called"] = True
        return "executed"

    registry.register(
        ToolDefinition(
            name="protected_tool",
            description="A protected tool.",
            handler=protected_tool,
            supported_intents={"tool_action"},
        )
    )

    result = run_orchestration(
        orchestrator,
        intent=make_intent(),
    )

    assert result.status == OrchestrationStatus.SELECTION_REQUIRED
    assert tracker["called"] is False
    assert result.verification is None

def test_lifecycle_events_follow_execution_and_verification_order(
    setup_orchestrator,
):
    registry, orchestrator, verification_engine = setup_orchestrator

    events = []

    def handler():
        events.append("handler")
        return "completed"

    def verifier(result):
        events.append("verifier")
        return result == "completed"

    def observer(state):
        events.append(state)

    registry.register(
        ToolDefinition(
            name="lifecycle_tool",
            description="Tests lifecycle ordering.",
            handler=handler,
            supported_intents={"tool_action"},
        )
    )

    verification_engine.register("lifecycle_tool", verifier)

    result = run_orchestration(
        orchestrator,
        intent=make_intent(),
        selected_tool="lifecycle_tool",
        on_state_change=observer,
    )

    assert result.status == OrchestrationStatus.VERIFICATION_SUCCEEDED

    assert events == [
        "executing",
        "handler",
        "verifying",
        "verifier",
    ]


def test_invalid_arguments_emit_no_execution_lifecycle_event(
    setup_orchestrator,
):
    registry, orchestrator, _ = setup_orchestrator

    events = []

    def observer(state):
        events.append(state)

    registry.register(
        ToolDefinition(
            name="validated_tool",
            description="Requires an integer.",
            handler=lambda value: value,
            parameters={
                "type": "object",
                "properties": {
                    "value": {"type": "integer"},
                },
                "required": ["value"],
                "additionalProperties": False,
            },
            supported_intents={"tool_action"},
        )
    )

    result = run_orchestration(
        orchestrator,
        intent=make_intent(),
        selected_tool="validated_tool",
        arguments={"value": "invalid"},
        on_state_change=observer,
    )

    assert result.status == OrchestrationStatus.EXECUTION_FAILED
    assert events == []


def test_verification_failure_never_emits_completed(
    setup_orchestrator,
):
    registry, orchestrator, verification_engine = setup_orchestrator

    events = []

    registry.register(
        ToolDefinition(
            name="unconfirmed_tool",
            description="Execution succeeds but verification fails.",
            handler=lambda: "executed",
            supported_intents={"tool_action"},
        )
    )

    verification_engine.register(
        "unconfirmed_tool",
        lambda result: False,
    )

    result = run_orchestration(
        orchestrator,
        intent=make_intent(),
        selected_tool="unconfirmed_tool",
        on_state_change=events.append,
    )

    assert result.status == OrchestrationStatus.VERIFICATION_FAILED
    assert events == ["executing", "verifying"]
    assert "completed" not in events
