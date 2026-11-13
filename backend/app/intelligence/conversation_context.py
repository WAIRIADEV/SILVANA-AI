
import re
from dataclasses import dataclass
from typing import Sequence

from backend.app.intelligence.base import ChatMessage


@dataclass(frozen=True)
class ConversationContext:
    """Structured context extracted from a conversation."""

    latest_user_message: str | None = None
    previous_user_message: str | None = None
    previous_assistant_message: str | None = None

    has_history: bool = False
    is_follow_up: bool = False
    is_correction: bool = False
    is_new_topic: bool = False
    references_previous_context: bool = False

    matched_rule: str | None = None


class ConversationContextAnalyzer:
    """
    Extracts basic conversational context using deterministic rules.

    This component does not classify the user's overall intent,
    execute tools, or make authorization decisions.
    """

    _FOLLOW_UP_PATTERNS = (
        r"\bmake it (?:shorter|simpler|clearer|longer)\b",
        r"\b(?:give|show) me (?:an example|another example)\b",
        r"\bexplain (?:that|it) (?:further|more|again)\b",
        r"\b(?:elaborate|expand) on (?:that|it)\b",
        r"\bcontinue\b",
        r"\bgo into more detail\b",
        r"\bwhat about (?:that|it|the other one)\b",
        r"\bwhy is that\b",
        r"\bdo the same\b",
        r"\bcan you simplify that\b",
        r"\bcan you explain that\b",
    )

    _CORRECTION_PATTERNS = (
        r"\bactually,?\s+i meant\b",
        r"\bi mean\b",
        r"\bcorrection\b",
        r"\bthat's not what i meant\b",
        r"\bthat is not what i meant\b",
        r"\bnot .+ but\b",
        r"\binstead,?\b",
    )

    _NEW_TOPIC_PATTERNS = (
        r"\bnew topic\b",
        r"\bunrelated question\b",
        r"\bswitching topics\b",
        r"\bchange of subject\b",
        r"\bon another subject\b",
    )

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"\s+", " ", text.lower().strip())

    def analyze(
        self,
        messages: Sequence[ChatMessage],
    ) -> ConversationContext:
        """
        Analyze conversation history and extract the latest context.

        Messages must be in chronological order.
        The latest user message is expected to be the current request.
        """

        if not messages:
            return ConversationContext()

        user_messages = [
            (index, message)
            for index, message in enumerate(messages)
            if message.role == "user"
        ]

        if not user_messages:
            return ConversationContext()

        latest_user_index, latest_user = user_messages[-1]
        latest_text = self._normalize(latest_user.content)

        previous_user = (
            user_messages[-2][1]
            if len(user_messages) >= 2
            else None
        )

        previous_assistant = next(
            (
                message
                for message in reversed(messages[:latest_user_index])
                if message.role == "assistant"
            ),
            None,
        )

        has_history = previous_assistant is not None

        if not has_history:
            return ConversationContext(
                latest_user_message=latest_user.content,
                has_history=False,
            )

        is_new_topic = any(
            re.search(pattern, latest_text)
            for pattern in self._NEW_TOPIC_PATTERNS
        )

        is_correction = any(
            re.search(pattern, latest_text)
            for pattern in self._CORRECTION_PATTERNS
        )

        is_follow_up = any(
            re.search(pattern, latest_text)
            for pattern in self._FOLLOW_UP_PATTERNS
        )

        if is_new_topic:
            matched_rule = "explicit_new_topic"
        elif is_correction:
            matched_rule = "correction"
        elif is_follow_up:
            matched_rule = "follow_up"
        else:
            matched_rule = None

        references_previous_context = (
            not is_new_topic
            and (is_follow_up or is_correction)
        )

        return ConversationContext(
            latest_user_message=latest_user.content,
            previous_user_message=(
                previous_user.content if previous_user else None
            ),
            previous_assistant_message=previous_assistant.content,
            has_history=True,
            is_follow_up=is_follow_up and not is_new_topic,
            is_correction=is_correction and not is_new_topic,
            is_new_topic=is_new_topic,
            references_previous_context=references_previous_context,
            matched_rule=matched_rule,
        )
