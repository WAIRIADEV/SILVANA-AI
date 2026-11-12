
import pytest

from backend.app.intelligence.tool_registry import (
    ToolDefinition,
    ToolRegistry,
)


@pytest.fixture
def registry():
    return ToolRegistry()


@pytest.fixture
def sample_tool():
    return ToolDefinition(
        name="calculate",
        description="Performs a basic calculation.",
        handler=lambda a, b: a + b,
    )


def test_register_tool_successfully(registry, sample_tool):
    registry.register(sample_tool)

    assert registry.get("calculate") is sample_tool


def test_duplicate_registration_is_rejected(
    registry,
    sample_tool,
):
    registry.register(sample_tool)

    with pytest.raises(ValueError, match="already registered"):
        registry.register(sample_tool)


def test_retrieve_unknown_tool_raises_error(registry):
    with pytest.raises(KeyError, match="not available"):
        registry.get("unknown_tool")


def test_disabled_tool_cannot_be_retrieved(registry):
    tool = ToolDefinition(
        name="disabled_tool",
        description="A disabled capability.",
        handler=lambda: None,
        enabled=False,
    )

    registry.register(tool)

    with pytest.raises(KeyError, match="not available"):
        registry.get("disabled_tool")


def test_list_tools_returns_enabled_tools_only(
    registry,
    sample_tool,
):
    disabled_tool = ToolDefinition(
        name="disabled_tool",
        description="A disabled capability.",
        handler=lambda: None,
        enabled=False,
    )

    registry.register(sample_tool)
    registry.register(disabled_tool)

    tools = registry.list_tools()

    assert len(tools) == 1
    assert tools[0].name == "calculate"


def test_unregister_tool_successfully(
    registry,
    sample_tool,
):
    registry.register(sample_tool)

    registry.unregister("calculate")

    with pytest.raises(KeyError, match="not available"):
        registry.get("calculate")


def test_unregister_unknown_tool_raises_error(registry):
    with pytest.raises(KeyError, match="not registered"):
        registry.unregister("unknown_tool")


def test_invalid_tool_type_is_rejected(registry):
    with pytest.raises(TypeError, match="ToolDefinition"):
        registry.register("not_a_tool")


def test_empty_tool_name_is_rejected():
    with pytest.raises(ValueError, match="name cannot be empty"):
        ToolDefinition(
            name="",
            description="Invalid tool.",
            handler=lambda: None,
        )


def test_empty_tool_description_is_rejected():
    with pytest.raises(ValueError, match="description cannot be empty"):
        ToolDefinition(
            name="invalid_tool",
            description="",
            handler=lambda: None,
        )


def test_non_callable_handler_is_rejected():
    with pytest.raises(ValueError, match="handler must be callable"):
        ToolDefinition(
            name="invalid_tool",
            description="Invalid handler.",
            handler="not_callable",
        )


def test_retrieving_tool_does_not_execute_handler(
    registry,
):
    execution_tracker = {"called": False}

    def sample_handler():
        execution_tracker["called"] = True

    tool = ToolDefinition(
        name="safe_tool",
        description="A tool used to verify registry behavior.",
        handler=sample_handler,
    )

    registry.register(tool)
    retrieved_tool = registry.get("safe_tool")

    assert retrieved_tool is tool
    assert execution_tracker["called"] is False
