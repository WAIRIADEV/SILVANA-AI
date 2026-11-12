
import asyncio

import pytest

from backend.app.intelligence.execution_engine import (
    ExecutionEngine,
    ExecutionResult,
)
from backend.app.intelligence.tool_registry import (
    ToolDefinition,
    ToolRegistry,
)
from backend.app.intelligence.tool_security import (
    ToolSecurityManager,
)


def run_execution(engine, tool_name, arguments, user_confirmed=False):
    """Run the asynchronous execution method inside a synchronous test."""

    return asyncio.run(
        engine.execute(
            tool_name=tool_name,
            arguments=arguments,
            user_confirmed=user_confirmed,
        )
    )


@pytest.fixture
def registry():
    return ToolRegistry()


@pytest.fixture
def security_manager():
    return ToolSecurityManager(
        granted_permissions={"read", "write"}
    )


@pytest.fixture
def engine(registry, security_manager):
    return ExecutionEngine(
        registry=registry,
        security_manager=security_manager,
    )


def test_sync_handler_executes_successfully(registry, engine):
    """A synchronous handler should execute and return its result."""

    def add_numbers(a, b):
        return a + b

    registry.register(
        ToolDefinition(
            name="add_numbers",
            description="Adds two numbers.",
            handler=add_numbers,
            parameters={
                "type": "object",
                "properties": {
                    "a": {"type": "integer"},
                    "b": {"type": "integer"},
                },
                "required": ["a", "b"],
                "additionalProperties": False,
            },
        )
    )

    result = run_execution(
        engine,
        "add_numbers",
        {"a": 10, "b": 20},
    )

    assert result.success is True
    assert result.result == 30
    assert result.error is None


def test_async_handler_executes_successfully(registry, engine):
    """An asynchronous handler should execute and return its result."""

    async def fetch_status():
        return {"status": "operational"}

    registry.register(
        ToolDefinition(
            name="fetch_status",
            description="Fetches system status.",
            handler=fetch_status,
        )
    )

    result = run_execution(
        engine,
        "fetch_status",
        {},
    )

    assert result.success is True
    assert result.result == {"status": "operational"}


def test_unknown_tool_is_rejected(engine):
    """Unregistered tools must never execute."""

    result = run_execution(
        engine,
        "unknown_tool",
        {},
    )

    assert result.success is False
    assert result.result is None
    assert result.error == "Requested tool is not available."


def test_disabled_tool_is_rejected(registry, engine):
    """Disabled tools must not execute."""

    registry.register(
        ToolDefinition(
            name="disabled_tool",
            description="A disabled tool.",
            handler=lambda: "executed",
            enabled=False,
        )
    )

    result = run_execution(
        engine,
        "disabled_tool",
        {},
    )

    assert result.success is False
    assert result.result is None
    assert result.error == "Requested tool is not available."


def test_invalid_arguments_are_rejected(registry, engine):
    """Invalid arguments must be rejected before handler invocation."""

    execution_tracker = {"called": False}

    def protected_tool(value):
        execution_tracker["called"] = True
        return value

    registry.register(
        ToolDefinition(
            name="protected_tool",
            description="A tool with validated arguments.",
            handler=protected_tool,
            parameters={
                "type": "object",
                "properties": {
                    "value": {"type": "integer"},
                },
                "required": ["value"],
                "additionalProperties": False,
            },
        )
    )

    result = run_execution(
        engine,
        "protected_tool",
        {"value": "not-an-integer"},
    )

    assert result.success is False
    assert result.error is not None
    assert execution_tracker["called"] is False


def test_missing_permission_is_rejected(registry, engine):
    """A tool must not execute without its required permission."""

    execution_tracker = {"called": False}

    def write_file():
        execution_tracker["called"] = True
        return "written"

    registry.register(
        ToolDefinition(
            name="write_file",
            description="Writes a file.",
            handler=write_file,
            permission_level="system",
        )
    )

    result = run_execution(
        engine,
        "write_file",
        {},
    )

    assert result.success is False
    assert "Permission" in result.error
    assert execution_tracker["called"] is False


def test_missing_confirmation_is_rejected(registry, engine):
    """Tools requiring confirmation must not execute without it."""

    execution_tracker = {"called": False}

    def send_message():
        execution_tracker["called"] = True
        return "sent"

    registry.register(
        ToolDefinition(
            name="send_message",
            description="Sends a message.",
            handler=send_message,
            requires_confirmation=True,
            permission_level="write",
        )
    )

    result = run_execution(
        engine,
        "send_message",
        {},
        user_confirmed=False,
    )

    assert result.success is False
    assert "confirmation" in result.error.lower()
    assert execution_tracker["called"] is False


def test_confirmed_tool_executes_successfully(registry, engine):
    """A permitted and confirmed tool should execute."""

    registry.register(
        ToolDefinition(
            name="send_message",
            description="Sends a message.",
            handler=lambda: "message sent",
            requires_confirmation=True,
            permission_level="write",
        )
    )

    result = run_execution(
        engine,
        "send_message",
        {},
        user_confirmed=True,
    )

    assert result.success is True
    assert result.result == "message sent"


def test_handler_exception_returns_safe_error(registry, engine):
    """Internal handler exceptions must not leak to the caller."""

    def failing_tool():
        raise RuntimeError("Sensitive internal implementation detail")

    registry.register(
        ToolDefinition(
            name="failing_tool",
            description="A tool that fails.",
            handler=failing_tool,
        )
    )

    result = run_execution(
        engine,
        "failing_tool",
        {},
    )

    assert result.success is False
    assert result.result is None
    assert result.error == "Tool execution failed."
    assert "Sensitive internal implementation detail" not in result.error


def test_execution_metadata_is_returned(registry, engine):
    """Execution results should contain traceable metadata."""

    registry.register(
        ToolDefinition(
            name="metadata_tool",
            description="Returns metadata test data.",
            handler=lambda: "complete",
        )
    )

    result = run_execution(
        engine,
        "metadata_tool",
        {},
    )

    assert isinstance(result, ExecutionResult)
    assert result.execution_id
    assert result.tool_name == "metadata_tool"
    assert isinstance(result.success, bool)
    assert isinstance(result.duration_ms, float)
    assert result.duration_ms >= 0


def test_each_execution_gets_a_unique_id(registry, engine):
    """Separate executions should receive separate execution IDs."""

    registry.register(
        ToolDefinition(
            name="identity_tool",
            description="Tests execution identity.",
            handler=lambda: "ok",
        )
    )

    first = run_execution(engine, "identity_tool", {})
    second = run_execution(engine, "identity_tool", {})

    assert first.success is True
    assert second.success is True
    assert first.execution_id != second.execution_id
