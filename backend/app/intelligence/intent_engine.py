
import re
from dataclasses import dataclass, replace
from enum import Enum

from backend.app.intelligence.conversation_context import (
    ConversationContext,
)


class Intent(str, Enum):
    GENERAL_CHAT = "general_chat"
    DATA_ANALYSIS = "data_analysis"
    DOCUMENT_QA = "document_qa"
    RESEARCH = "research"
    TOOL_ACTION = "tool_action"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class IntentResult:
    intent: Intent
    confidence: float
    requires_tool: bool
    requires_confirmation: bool
    matched_rule: str | None = None

    # Conversation context metadata.
    is_follow_up: bool = False
    is_correction: bool = False
    is_new_topic: bool = False
    references_previous_context: bool = False
    context_rule: str | None = None


class IntentEngine:
    """
    Baseline rule-based intent classifier.

    Confidence values are heuristic scores, not calibrated
    probabilities. This engine classifies requests only.

    Optional conversation context enriches the classification
    result but does not independently change its intent.
    """

    _RULES = [
        (
            Intent.TOOL_ACTION,
            0.92,
            "explicit_action",
            (
                r"\b(remind me|set a reminder)\b",
                r"\b(i need a reminder|i want a reminder)\b",
                r"\b(don't let me forget|do not let me forget)\b",
                r"\b(make sure i remember)\b",

                r"\b(remind me|set a reminder|schedule|book|send|"
                r"delete|remove|rename|move|copy|create|make|open|"
                r"close|download|upload|install|launch)\b.*"
                r"\b(folder|file|email|message|reminder|meeting|"
                r"appointment|calendar event|application|app|"
                r"browser|workflow)\b",

                r"\b(turn on|turn off|enable|disable)\b.*"
                r"\b(application|app|service|feature|device)\b",
            ),
        ),
        (
            Intent.DATA_ANALYSIS,
            0.90,
            "data_analysis",
            (
                r"\b(csv|dataframe|dataset|spreadsheet)\b",
                r"\b(correlation|regression|classification|"
                r"clustering|standard deviation|hypothesis test)\b",
                r"\b(analy[sz]e|clean|preprocess|visuali[sz]e|"
                r"plot|forecast|calculate|compute)\b.*"
                r"\b(data|dataset|csv|spreadsheet|records|"
                r"sales|metrics|numbers)\b",
            ),
        ),
        (
            Intent.DOCUMENT_QA,
            0.88,
            "document_question",
            (
                r"\b(pdf|docx|document|report|research paper|"
                r"contract)\b",
                r"\b(summarize|summarise|extract|quote)\b.*"
                r"\b(file|document|report|paper|pdf)\b",
                r"\b(what does|what is in|according to)\b.*"
                r"\b(document|report|pdf|file)\b",
            ),
        ),
        (
            Intent.RESEARCH,
            0.86,
            "research_request",
            (
                r"\b(research|investigate|look up|find sources|"
                r"latest developments|current developments)\b",
                r"\b(latest|recent|current)\b.*"
                r"\b(technology|news|developments|trends|"
                r"research|companies|models)\b",
            ),
        ),
    ]

    def classify(
        self,
        text: str,
        context: ConversationContext | None = None,
    ) -> IntentResult:
        """
        Classify a user request and optionally attach conversation context.
        """

        result = self._classify_message(text)

        if context is None:
            return result

        return replace(
            result,
            is_follow_up=context.is_follow_up,
            is_correction=context.is_correction,
            is_new_topic=context.is_new_topic,
            references_previous_context=(
                context.references_previous_context
            ),
            context_rule=context.matched_rule,
        )

    def _classify_message(self, text: str) -> IntentResult:
        """Apply the existing deterministic intent rules."""

        if not isinstance(text, str) or not text.strip():
            return IntentResult(
                intent=Intent.UNKNOWN,
                confidence=0.0,
                requires_tool=False,
                requires_confirmation=False,
            )

        normalized_text = re.sub(
            r"\s+",
            " ",
            text.lower().strip(),
        )

        for intent, confidence, rule_name, patterns in self._RULES:
            for pattern in patterns:
                if re.search(pattern, normalized_text):
                    return IntentResult(
                        intent=intent,
                        confidence=confidence,
                        requires_tool=True,
                        requires_confirmation=(
                            intent == Intent.TOOL_ACTION
                        ),
                        matched_rule=rule_name,
                    )

        if re.search(
            r"\b(hi|hello|hey|good morning|good evening)\b",
            normalized_text,
        ):
            return IntentResult(
                intent=Intent.GENERAL_CHAT,
                confidence=0.95,
                requires_tool=False,
                requires_confirmation=False,
                matched_rule="greeting",
            )

        return IntentResult(
            intent=Intent.GENERAL_CHAT,
            confidence=0.65,
            requires_tool=False,
            requires_confirmation=False,
            matched_rule="default_conversation",
        )
