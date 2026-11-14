
import pytest

from backend.app.intelligence.capability_resolver import (
    CapabilityResolver,
)
from backend.app.intelligence.intent_engine import (
    Intent,
    IntentEngine,
)
from backend.app.intelligence.tool_registry import (
    ToolDefinition,
    ToolRegistry,
)


@pytest.fixture
def registry():
    return ToolRegistry()


@pytest.fixture
def resolver(registry):
    return CapabilityResolver(registry)


def test_resolves_matching_intent(registry, resolver):
    tool = ToolDefinition(
        name="analyze_dataset",
        description="Analyzes datasets.",
        handler=lambda: None,
        supported_intents={"data_analysis"},
    )

    registry.register(tool)

    matches = resolver.resolve(Intent.DATA_ANALYSIS)

    assert len(matches) == 1
    assert matches[0].tool_name == "analyze_dataset"


def test_resolves_intent_result(registry, resolver):
    tool = ToolDefinition(
        name="research",
        description="Researches topics.",
        handler=lambda: None,
        supported_intents={"research"},
    )

    registry.register(tool)

    intent_result = IntentEngine().classify(
        "Research the latest AI developments."
    )

    matches = resolver.resolve(intent_result)

    assert len(matches) == 1
    assert matches[0].tool_name == "research"


def test_returns_multiple_matching_capabilities(
    registry,
    resolver,
):
    registry.register(
        ToolDefinition(
            name="tool_one",
            description="First tool.",
            handler=lambda: None,
            supported_intents={"data_analysis"},
        )
    )

    registry.register(
        ToolDefinition(
            name="tool_two",
            description="Second tool.",
            handler=lambda: None,
            supported_intents={"data_analysis"},
        )
    )

    matches = resolver.resolve(Intent.DATA_ANALYSIS)

    assert len(matches) == 2


def test_unrelated_tools_are_not_returned(
    registry,
    resolver,
):
    registry.register(
        ToolDefinition(
            name="research",
            description="Research tool.",
            handler=lambda: None,
            supported_intents={"research"},
        )
    )

    matches = resolver.resolve(Intent.DATA_ANALYSIS)

    assert matches == []


def test_disabled_tools_are_not_returned(
    registry,
    resolver,
):
    registry.register(
        ToolDefinition(
            name="disabled_tool",
            description="Disabled capability.",
            handler=lambda: None,
            enabled=False,
            supported_intents={"data_analysis"},
        )
    )

    matches = resolver.resolve(Intent.DATA_ANALYSIS)

    assert matches == []


@pytest.mark.parametrize(
    "intent",
    [
        Intent.GENERAL_CHAT,
        Intent.UNKNOWN,
    ],
)
def test_non_tool_intents_return_no_capabilities(
    resolver,
    intent,
):
    assert resolver.resolve(intent) == []


def test_discovery_never_executes_handler(
    registry,
    resolver,
):
    execution_tracker = {"called": False}

    def handler():
        execution_tracker["called"] = True

    registry.register(
        ToolDefinition(
            name="safe_tool",
            description="A discovery test tool.",
            handler=handler,
            supported_intents={"tool_action"},
        )
    )

    resolver.resolve(Intent.TOOL_ACTION)

    assert execution_tracker["called"] is False


def test_invalid_intent_input_is_rejected(resolver):
    with pytest.raises(TypeError, match="Intent or IntentResult"):
        resolver.resolve("data_analysis")


def test_match_contains_permission_metadata(
    registry,
    resolver,
):
    registry.register(
        ToolDefinition(
            name="delete_file",
            description="Deletes a file.",
            handler=lambda: None,
            permission_level="destructive",
            requires_confirmation=True,
            supported_intents={"tool_action"},
        )
    )

    matches = resolver.resolve(Intent.TOOL_ACTION)

    assert matches[0].permission_level == "destructive"
    assert matches[0].requires_confirmation is True
