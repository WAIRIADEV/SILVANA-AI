
import pytest

from backend.app.intelligence.intent_engine import (
    Intent,
    IntentEngine,
)


@pytest.fixture
def engine():
    return IntentEngine()


@pytest.mark.parametrize(
    ("message", "expected_intent"),
    [
        ("Hello SILVANA", Intent.GENERAL_CHAT),
        ("Explain neural networks", Intent.GENERAL_CHAT),

        (
            "Analyze this sales dataset",
            Intent.DATA_ANALYSIS,
        ),
        (
            "Calculate the correlation in this CSV",
            Intent.DATA_ANALYSIS,
        ),

        (
            "Summarize this PDF",
            Intent.DOCUMENT_QA,
        ),
        (
            "What does this report say?",
            Intent.DOCUMENT_QA,
        ),

        (
            "Research current AI developments",
            Intent.RESEARCH,
        ),
        (
            "Find the latest machine learning trends",
            Intent.RESEARCH,
        ),

        (
            "Create a folder called Projects",
            Intent.TOOL_ACTION,
        ),
        (
            "Remind me to study Python tomorrow",
            Intent.TOOL_ACTION,
        ),
        (
            "I need a reminder to study tomorrow",
            Intent.TOOL_ACTION,
        ),
        (
            "Don't let me forget to submit my assignment",
            Intent.TOOL_ACTION,
        ),
        (
            "Make sure I remember my meeting tomorrow",
            Intent.TOOL_ACTION,
        ),
    ],
)
def test_classifies_common_intents(
    engine,
    message,
    expected_intent,
):
    result = engine.classify(message)

    assert result.intent == expected_intent


def test_empty_message_returns_unknown(engine):
    result = engine.classify("   ")

    assert result.intent == Intent.UNKNOWN
    assert result.confidence == 0.0
    assert result.requires_tool is False


def test_tool_action_requires_confirmation(engine):
    result = engine.classify(
        "Create a folder called Projects"
    )

    assert result.intent == Intent.TOOL_ACTION
    assert result.requires_tool is True
    assert result.requires_confirmation is True


def test_data_analysis_requires_tool(engine):
    result = engine.classify(
        "Analyze this sales dataset"
    )

    assert result.intent == Intent.DATA_ANALYSIS
    assert result.requires_tool is True
    assert result.requires_confirmation is False


def test_general_chat_does_not_require_tool(engine):
    result = engine.classify(
        "Explain how neural networks work"
    )

    assert result.intent == Intent.GENERAL_CHAT
    assert result.requires_tool is False
    assert result.requires_confirmation is False


def test_classification_is_case_insensitive(engine):
    result = engine.classify(
        "ANALYZE THIS SALES DATASET"
    )

    assert result.intent == Intent.DATA_ANALYSIS


def test_tool_action_takes_precedence(engine):
    result = engine.classify(
        "Create a folder for my research documents"
    )

    assert result.intent == Intent.TOOL_ACTION


def test_result_contains_matching_rule(engine):
    result = engine.classify(
        "Research current AI developments"
    )

    assert result.matched_rule == "research_request"
    assert result.confidence > 0.0
