
import re

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from backend.app.intelligence.capability_resolver import (
    CapabilityMatch,
)
from backend.app.intelligence.intent_engine import (
    Intent,
    IntentResult,
)


class ToolPlanStatus(str, Enum):
    READY = "ready"
    CLARIFICATION_REQUIRED = "clarification_required"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class ToolPlan:
    """
    A proposed tool operation.

    A plan describes what should be executed. It does not execute
    the operation or grant permission to execute it.
    """

    status: ToolPlanStatus
    tool_name: str | None = None
    arguments: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    message: str = ""
    matched_rule: str | None = None


class ToolPlanner:
    """
    Deterministically maps recognized filesystem requests to tools.

    The planner:
    - never accesses tool handlers;
    - never executes tools;
    - never grants permissions;
    - only proposes tools present in supplied capabilities;
    - checks proposed arguments against declared tool schemas.
    """

    _TOOL_BY_RULE = {
        "filesystem_metadata": "get_file_metadata",
        "filesystem_existence": "check_file_exists",
    }

    _FILE_PATTERN = re.compile(
        r"(?<![\w.-])"
        r"(?:[A-Za-z0-9_.-]+[\\/])*"
        r"[A-Za-z0-9_.-]+\.[A-Za-z0-9]{1,10}"
        r"(?![\w.-])"
    )

    _GLOB_PATTERN = re.compile(
        r"(?<![\w])"
        r"(?:\*|[A-Za-z0-9_-]+)"
        r"[A-Za-z0-9_*?.-]*"
        r"\.[A-Za-z0-9_*?-]{1,10}"
    )

    _QUOTED_PATH_PATTERN = re.compile(
        r"""["'`](?P<path>[^"'`]+)["'`]"""
    )

    _DIRECTORY_PATTERN = re.compile(
        r"\b(?:in|inside|under|within|of|from)\s+"
        r"(?:(?:the|my)\s+)?"
        r"(?:(?:folder|directory)\s+)?"
        r"(?P<path>[A-Za-z0-9_.-]+(?:[\\/][A-Za-z0-9_.-]+)*)",
        re.IGNORECASE,
    )

    _EXTENSION_MAP = {
        "python": "*.py",
        "pdf": "*.pdf",
        "word": "*.docx",
        "excel": "*.xlsx",
        "csv": "*.csv",
        "json": "*.json",
        "javascript": "*.js",
        "typescript": "*.ts",
        "text": "*.txt",
    }

    _WORKSPACE_WORDS = {
        "workspace",
        "workspaces",
        "here",
        "root",
    }

    def plan(
        self,
        intent: IntentResult,
        user_request: str,
        available_tools: list[CapabilityMatch],
    ) -> ToolPlan:
        """
        Produce a tool proposal without executing it.
        """

        if not isinstance(intent, IntentResult):
            raise TypeError("Expected an IntentResult.")

        if not isinstance(user_request, str) or not user_request.strip():
            return self._clarification(
                "Please describe what you want SILVANA to inspect."
            )

        if not isinstance(available_tools, list):
            raise TypeError("Available tools must be supplied as a list.")

        if intent.intent != Intent.TOOL_ACTION:
            return self._unsupported(
                "This request does not require a supported filesystem tool."
            )

        tool_name = self._select_tool(
            intent.matched_rule,
            user_request,
        )

        if intent.matched_rule == "explicit_action":
            return self._unsupported(
                "That action is not supported by SILVANA's current filesystem capabilities."
            )

        if tool_name is None:
            return self._unsupported(
                "I could not identify a supported filesystem operation."
            )

        capability = self._find_capability(
            tool_name,
            available_tools,
        )

        if capability is None:
            return self._unsupported(
                "The required filesystem capability is not currently available."
            )

        normalized_request = re.sub(
            r"\s+",
            " ",
            user_request.strip(),
        )

        arguments = self._build_arguments(
            tool_name,
            normalized_request,
        )

        if arguments is None:
            return self._clarification(
                self._clarification_message(tool_name),
                matched_rule=intent.matched_rule,
            )

        if not self._arguments_match_schema(
            arguments,
            capability.parameters,
        ):
            return self._unsupported(
                "The proposed operation does not match the registered tool schema.",
                matched_rule=intent.matched_rule,
            )

        return ToolPlan(
            status=ToolPlanStatus.READY,
            tool_name=tool_name,
            arguments=arguments,
            confidence=0.95,
            message="A supported filesystem operation has been identified.",
            matched_rule=intent.matched_rule,
        )

    @staticmethod
    def _select_tool(
        matched_rule: str | None,
        request: str,
    ) -> str | None:
        if matched_rule == "filesystem_search":
            normalized = request.lower()

            if re.search(
                r"\b(list|show|display)\b",
                normalized,
            ):
                return "list_directory"

            return "search_files"

        return ToolPlanner._TOOL_BY_RULE.get(matched_rule)

    @staticmethod
    def _find_capability(
        tool_name: str,
        available_tools: list[CapabilityMatch],
    ) -> CapabilityMatch | None:
        for capability in available_tools:
            if not isinstance(capability, CapabilityMatch):
                continue

            if capability.tool_name == tool_name:
                return capability

        return None

    def _build_arguments(
        self,
        tool_name: str,
        request: str,
    ) -> dict[str, Any] | None:

        if tool_name == "list_directory":
            return {
                "path": self._extract_directory(request) or ".",
            }

        if tool_name == "search_files":
            pattern = self._extract_search_pattern(request)
            directory = self._extract_directory(request)

            arguments = {}

            if pattern:
                arguments["pattern"] = pattern

            if directory:
                arguments["directory"] = directory

            return arguments

        if tool_name in {
            "get_file_metadata",
            "check_file_exists",
        }:
            target = self._extract_target(request)

            if not target:
                return None

            return {"path": target}

        return None

    def _extract_search_pattern(
        self,
        request: str,
    ) -> str | None:
        glob_match = self._GLOB_PATTERN.search(request)

        if glob_match:
            return glob_match.group(0)

        file_match = self._FILE_PATTERN.search(request)

        if file_match:
            return (
                file_match.group(0)
                .replace("\\", "/")
                .split("/")[-1]
            )

        normalized = request.lower()

        for keyword, pattern in self._EXTENSION_MAP.items():
            if re.search(
                rf"\b{re.escape(keyword)}\b",
                normalized,
            ):
                return pattern

        return None

    def _extract_target(
        self,
        request: str,
    ) -> str | None:
        quoted_match = self._QUOTED_PATH_PATTERN.search(request)

        if quoted_match:
            target = quoted_match.group("path").strip()

            if target:
                return target

        file_match = self._FILE_PATTERN.search(request)

        if file_match:
            return file_match.group(0).replace("\\", "/")

        directory_match = self._DIRECTORY_PATTERN.search(request)

        if directory_match:
            target = directory_match.group("path").strip()

            if target.lower() not in self._WORKSPACE_WORDS:
                return target

        return None

    def _extract_directory(
        self,
        request: str,
    ) -> str | None:
        quoted_match = self._QUOTED_PATH_PATTERN.search(request)

        if quoted_match:
            target = quoted_match.group("path").strip()

            if target and target.lower() not in self._WORKSPACE_WORDS:
                return target

        file_match = self._FILE_PATTERN.search(request)

        if file_match:
            file_path = file_match.group(0).replace("\\", "/")

            if "/" in file_path:
                return file_path.rsplit("/", 1)[0]

        directory_match = self._DIRECTORY_PATTERN.search(request)

        if directory_match:
            target = directory_match.group("path").strip()

            if target.lower() not in self._WORKSPACE_WORDS:
                return target

        return None

    @staticmethod
    def _arguments_match_schema(
        arguments: dict[str, Any],
        schema: dict[str, Any],
    ) -> bool:
        """
        Perform a basic planner-side schema consistency check.

        ExecutionEngine remains responsible for full validation.
        """

        if not isinstance(schema, dict):
            return False

        properties = schema.get("properties", {})
        required = schema.get("required", [])

        if not isinstance(properties, dict):
            return False

        if not isinstance(required, list):
            return False

        if any(key not in properties for key in arguments):
            return False

        if any(key not in arguments for key in required):
            return False

        return True

    @staticmethod
    def _clarification_message(
        tool_name: str,
    ) -> str:
        if tool_name == "get_file_metadata":
            return "Which file or folder should I inspect?"

        if tool_name == "check_file_exists":
            return "Which file or folder should I check?"

        return "Could you specify the file or folder you want me to inspect?"

    @staticmethod
    def _clarification(
        message: str,
        matched_rule: str | None = None,
    ) -> ToolPlan:
        return ToolPlan(
            status=ToolPlanStatus.CLARIFICATION_REQUIRED,
            message=message,
            matched_rule=matched_rule,
        )

    @staticmethod
    def _unsupported(
        message: str,
        matched_rule: str | None = None,
    ) -> ToolPlan:
        return ToolPlan(
            status=ToolPlanStatus.UNSUPPORTED,
            message=message,
            matched_rule=matched_rule,
        )
