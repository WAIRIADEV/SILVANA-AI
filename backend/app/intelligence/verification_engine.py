
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable


class VerificationStatus(str, Enum):
    VERIFIED = "verified"
    FAILED = "failed"
    UNVERIFIABLE = "unverifiable"


@dataclass(frozen=True)
class VerificationResult:
    execution_id: str
    tool_name: str
    status: VerificationStatus
    verified: bool
    message: str
    evidence: Any = None


VerificationHandler = Callable[[Any], bool]


class VerificationEngine:
    """
    Independently evaluates whether a tool execution achieved
    its intended outcome.

    A missing verification handler does not imply success.
    """

    def __init__(self):
        self._verifiers: dict[str, VerificationHandler] = {}

    def register(
        self,
        tool_name: str,
        verifier: VerificationHandler,
    ) -> None:
        if not isinstance(tool_name, str) or not tool_name.strip():
            raise ValueError("Tool name cannot be empty.")

        if not callable(verifier):
            raise ValueError("Verifier must be callable.")

        normalized_name = tool_name.strip()

        if normalized_name in self._verifiers:
            raise ValueError(
                f"A verifier for '{normalized_name}' is already registered."
            )

        self._verifiers[normalized_name] = verifier

    def unregister(self, tool_name: str) -> None:
        if not isinstance(tool_name, str) or not tool_name.strip():
            raise ValueError("Tool name cannot be empty.")

        normalized_name = tool_name.strip()

        if normalized_name not in self._verifiers:
            raise KeyError(
                f"No verifier is registered for '{normalized_name}'."
            )

        del self._verifiers[normalized_name]

    def verify(
        self,
        execution_id: str,
        tool_name: str,
        execution_success: bool,
        execution_result: Any = None,
    ) -> VerificationResult:

        if not isinstance(execution_id, str) or not execution_id.strip():
            raise ValueError("Execution ID cannot be empty.")

        if not isinstance(tool_name, str) or not tool_name.strip():
            raise ValueError("Tool name cannot be empty.")

        if not isinstance(execution_success, bool):
            raise ValueError("Execution success must be a boolean.")

        normalized_id = execution_id.strip()
        normalized_name = tool_name.strip()

        if not execution_success:
            return VerificationResult(
                execution_id=normalized_id,
                tool_name=normalized_name,
                status=VerificationStatus.FAILED,
                verified=False,
                message="Execution failed; outcome cannot be verified.",
            )

        verifier = self._verifiers.get(normalized_name)

        if verifier is None:
            return VerificationResult(
                execution_id=normalized_id,
                tool_name=normalized_name,
                status=VerificationStatus.UNVERIFIABLE,
                verified=False,
                message="No verification procedure is registered for this tool.",
            )

        try:
            evidence = verifier(execution_result)
        except Exception:
            return VerificationResult(
                execution_id=normalized_id,
                tool_name=normalized_name,
                status=VerificationStatus.UNVERIFIABLE,
                verified=False,
                message="The verification procedure could not be completed.",
            )

        if not isinstance(evidence, bool):
            return VerificationResult(
                execution_id=normalized_id,
                tool_name=normalized_name,
                status=VerificationStatus.UNVERIFIABLE,
                verified=False,
                message="The verification procedure returned an invalid result.",
                evidence=evidence,
            )

        if evidence:
            return VerificationResult(
                execution_id=normalized_id,
                tool_name=normalized_name,
                status=VerificationStatus.VERIFIED,
                verified=True,
                message="The intended outcome was independently verified.",
                evidence=evidence,
            )

        return VerificationResult(
            execution_id=normalized_id,
            tool_name=normalized_name,
            status=VerificationStatus.FAILED,
            verified=False,
            message="The intended outcome could not be confirmed.",
            evidence=evidence,
        )
