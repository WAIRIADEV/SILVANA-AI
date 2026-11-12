
import re
from dataclasses import dataclass
from enum import Enum


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


class IntentEngine:
    """
    Baseline rule-based intent classifier.

    Confidence values are heuristic scores, not calibrated
    probabilities. This engine classifies requests only;
    it does not execute tools or actions.
    """

    _RULES = [
        (
            Intent.TOOL_ACTION,
            0.92,
            "explicit_action",
            (
                # Reminder requests.
                r"\b(remind me|set a reminder)\b",
                r"\b(i need a reminder|i want a reminder)\b",
                r"\b(don't let me forget|do not let me forget)\b",
                r"\b(make sure i remember)\b",

                # Explicit actions involving identifiable objects.
                r"\b(remind me|set a reminder|schedule|book|send|"
                r"delete|remove|rename|move|copy|create|make|open|"
                r"close|download|upload|install|launch)\b.*"
                r"\b(folder|file|email|message|reminder|meeting|"
                r"appointment|calendar event|application|app|"
                r"browser|workflow)\b",

                # Application and service controls.
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

    def classify(self, text: str) -> IntentResult:
        """
        Classify a user request using ordered keyword rules.
        """

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
