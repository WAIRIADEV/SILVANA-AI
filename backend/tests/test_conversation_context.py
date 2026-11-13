
import pytest

from backend.app.intelligence.base import ChatMessage
from backend.app.intelligence.conversation_context import (
    ConversationContextAnalyzer,
)


@pytest.fixture
def analyzer():
    return ConversationContextAnalyzer()


def make_conversation(*messages):
    return [
        ChatMessage(role=role, content=content)
        for role, content in messages
    ]


def test_empty_conversation_returns_empty_context(analyzer):
    result = analyzer.analyze([])

    assert result.latest_user_message is None
    assert result.has_history is False


def test_identifies_latest_user_message(analyzer):
    messages = make_conversation(
        ("user", "Explain regression."),
        ("assistant", "Regression models relationships."),
        ("user", "Give me an example."),
    )

    result = analyzer.analyze(messages)

    assert result.latest_user_message == "Give me an example."


def test_extracts_previous_exchange(analyzer):
    messages = make_conversation(
        ("user", "Explain regression."),
        ("assistant", "Regression models relationships."),
        ("user", "Give me an example."),
    )

    result = analyzer.analyze(messages)

    assert result.previous_user_message == "Explain regression."
    assert result.previous_assistant_message == (
        "Regression models relationships."
    )


@pytest.mark.parametrize(
    "message",
    [
        "Make it shorter.",
        "Make it simpler.",
        "Give me an example.",
        "Explain that further.",
        "Can you simplify that?",
        "Go into more detail.",
        "Do the same.",
    ],
)
def test_detects_follow_up_requests(analyzer, message):
    messages = make_conversation(
        ("user", "Explain regression."),
        ("assistant", "Regression models relationships."),
        ("user", message),
    )

    result = analyzer.analyze(messages)

    assert result.is_follow_up is True
    assert result.references_previous_context is True
    assert result.matched_rule == "follow_up"


@pytest.mark.parametrize(
    "message",
    [
        "Actually, I meant classification.",
        "I mean deep learning.",
        "Correction: use the second dataset.",
        "That's not what I meant.",
    ],
)
def test_detects_corrections(analyzer, message):
    messages = make_conversation(
        ("user", "Explain regression."),
        ("assistant", "Regression models relationships."),
        ("user", message),
    )

    result = analyzer.analyze(messages)

    assert result.is_correction is True
    assert result.references_previous_context is True
    assert result.matched_rule == "correction"


@pytest.mark.parametrize(
    "message",
    [
        "New topic: explain computer vision.",
        "Switching topics, tell me about Python.",
        "Unrelated question: what is an API?",
    ],
)
def test_detects_explicit_new_topics(analyzer, message):
    messages = make_conversation(
        ("user", "Explain regression."),
        ("assistant", "Regression models relationships."),
        ("user", message),
    )

    result = analyzer.analyze(messages)

    assert result.is_new_topic is True
    assert result.is_follow_up is False
    assert result.is_correction is False
    assert result.references_previous_context is False


def test_first_user_message_has_no_previous_context(analyzer):
    messages = make_conversation(
        ("user", "Explain regression."),
    )

    result = analyzer.analyze(messages)

    assert result.has_history is False
    assert result.latest_user_message == "Explain regression."
    assert result.previous_assistant_message is None


def test_ordinary_new_question_is_not_assumed_to_be_follow_up(analyzer):
    messages = make_conversation(
        ("user", "Explain regression."),
        ("assistant", "Regression models relationships."),
        ("user", "How does computer vision work?"),
    )

    result = analyzer.analyze(messages)

    assert result.has_history is True
    assert result.is_follow_up is False
    assert result.is_correction is False
    assert result.is_new_topic is False
    assert result.references_previous_context is False


def test_ignores_system_messages(analyzer):
    messages = make_conversation(
        ("system", "You are SILVANA."),
        ("user", "Explain regression."),
        ("assistant", "Regression models relationships."),
        ("user", "Make it shorter."),
    )

    result = analyzer.analyze(messages)

    assert result.latest_user_message == "Make it shorter."
    assert result.is_follow_up is True
