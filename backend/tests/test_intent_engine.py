
import pytest

from backend.app.intelligence.base import ChatMessage
from backend.app.intelligence.conversation_context import (
    ConversationContextAnalyzer,
)
from backend.app.intelligence.intent_engine import (
    Intent,
    IntentEngine,
)


@pytest.fixture
def engine():
    return IntentEngine()


@pytest.fixture
def analyzer():
    return ConversationContextAnalyzer()


def make_conversation(*messages):
    return [
        ChatMessage(role=role, content=content)
        for role, content in messages
    ]


# ---------------------------------------------------------
# BASIC INTENT CLASSIFICATION
# ---------------------------------------------------------

@pytest.mark.parametrize(
    ("message", "expected_intent"),
    [
        ("Hello SILVANA", Intent.GENERAL_CHAT),
        ("Explain neural networks", Intent.GENERAL_CHAT),

        ("Analyze this sales dataset", Intent.DATA_ANALYSIS),
        ("Calculate the correlation in this CSV", Intent.DATA_ANALYSIS),

        ("Summarize this PDF", Intent.DOCUMENT_QA),
        ("What does this report say?", Intent.DOCUMENT_QA),

        ("Research current AI developments", Intent.RESEARCH),
        ("Find the latest machine learning trends", Intent.RESEARCH),

        ("Create a folder called Projects", Intent.TOOL_ACTION),
        ("Remind me to study Python tomorrow", Intent.TOOL_ACTION),
        ("I need a reminder to study tomorrow", Intent.TOOL_ACTION),
        ("Don't let me forget to submit my assignment", Intent.TOOL_ACTION),
        ("Make sure I remember my meeting tomorrow", Intent.TOOL_ACTION),
    ],
)
def test_classifies_common_intents(
    engine,
    message,
    expected_intent,
):
    result = engine.classify(message)

    assert result.intent == expected_intent


# ---------------------------------------------------------
# INTENT RESULT VALIDATION
# ---------------------------------------------------------

def test_empty_message_returns_unknown(engine):
    result = engine.classify("   ")

    assert result.intent == Intent.UNKNOWN
    assert result.confidence == 0.0
    assert result.requires_tool is False
    assert result.requires_confirmation is False


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


# ---------------------------------------------------------
# CONVERSATION CONTEXT INTEGRATION
# ---------------------------------------------------------

def test_intent_engine_accepts_conversation_context(
    engine,
    analyzer,
):
    messages = make_conversation(
        ("user", "Explain regression."),
        (
            "assistant",
            "Regression models relationships between variables.",
        ),
        ("user", "Give me an example."),
    )

    context = analyzer.analyze(messages)

    result = engine.classify(
        "Give me an example.",
        context,
    )

    assert result.intent == Intent.GENERAL_CHAT
    assert result.is_follow_up is True
    assert result.references_previous_context is True
    assert result.context_rule == "follow_up"


def test_context_does_not_override_intent_classification(
    engine,
    analyzer,
):
    messages = make_conversation(
        ("user", "Explain regression."),
        ("assistant", "Regression models relationships."),
        ("user", "Create a folder called Projects."),
    )

    context = analyzer.analyze(messages)

    result = engine.classify(
        "Create a folder called Projects.",
        context,
    )

    assert result.intent == Intent.TOOL_ACTION
    assert result.requires_confirmation is True


def test_intent_engine_works_without_context(engine):
    result = engine.classify(
        "Explain neural networks."
    )

    assert result.intent == Intent.GENERAL_CHAT
    assert result.is_follow_up is False
    assert result.is_correction is False
    assert result.is_new_topic is False
    assert result.references_previous_context is False
    assert result.context_rule is None


def test_context_correction_is_preserved(
    engine,
    analyzer,
):
    messages = make_conversation(
        ("user", "Explain regression."),
        ("assistant", "Regression models relationships."),
        ("user", "Actually, I meant classification."),
    )

    context = analyzer.analyze(messages)

    result = engine.classify(
        "Actually, I meant classification.",
        context,
    )

    assert result.is_correction is True
    assert result.references_previous_context is True
    assert result.context_rule == "correction"


def test_new_topic_context_is_preserved(
    engine,
    analyzer,
):
    messages = make_conversation(
        ("user", "Explain regression."),
        ("assistant", "Regression models relationships."),
        ("user", "New topic: explain computer vision."),
    )

    context = analyzer.analyze(messages)

    result = engine.classify(
        "New topic: explain computer vision.",
        context,
    )

    assert result.is_new_topic is True
    assert result.is_follow_up is False
    assert result.references_previous_context is False
    assert result.context_rule == "explicit_new_topic"


def test_context_does_not_change_original_confidence(
    engine,
    analyzer,
):
    messages = make_conversation(
        ("user", "Explain regression."),
        ("assistant", "Regression models relationships."),
        ("user", "Give me an example."),
    )

    context = analyzer.analyze(messages)

    without_context = engine.classify(
        "Give me an example."
    )

    with_context = engine.classify(
        "Give me an example.",
        context,
    )

    assert with_context.confidence == without_context.confidence
    assert with_context.intent == without_context.intent
