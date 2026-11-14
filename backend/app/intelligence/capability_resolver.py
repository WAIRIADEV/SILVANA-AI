
from dataclasses import dataclass
from typing import Any

from backend.app.intelligence.intent_engine import (
    Intent,
    IntentResult,
)
from backend.app.intelligence.tool_registry import ToolRegistry


@dataclass(frozen=True)
class CapabilityMatch:
    """
    Safe metadata describing a tool matching a user intent.

    Does not expose or execute the registered handler.
    """

    tool_name: str
    description: str
    parameters: dict[str, Any]
    permission_level: str
    requires_confirmation: bool


class CapabilityResolver:
    """
    Discovers enabled tools capable of handling an intent.

    Discovery does not authorize or execute tools.
    """

    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    def resolve(
        self,
        intent: Intent | IntentResult,
    ) -> list[CapabilityMatch]:
        """
        Return enabled tools explicitly supporting the given intent.
        """

        if isinstance(intent, IntentResult):
            intent_value = intent.intent.value

        elif isinstance(intent, Intent):
            intent_value = intent.value

        else:
            raise TypeError(
                "Expected an Intent or IntentResult."
            )

        if intent_value in {
            Intent.GENERAL_CHAT.value,
            Intent.UNKNOWN.value,
        }:
            return []

        matches = []

        for tool in self.registry.list_tools():
            if intent_value not in tool.supported_intents:
                continue

            matches.append(
                CapabilityMatch(
                    tool_name=tool.name,
                    description=tool.description,
                    parameters=dict(tool.parameters),
                    permission_level=tool.permission_level,
                    requires_confirmation=tool.requires_confirmation,
                )
            )

        return matches
