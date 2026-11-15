
import json

from dataclasses import dataclass

from backend.app.intelligence.execution_engine import ExecutionResult
from backend.app.intelligence.tool_orchestrator import (
    OrchestrationResult,
    OrchestrationStatus,
)
from backend.app.intelligence.verification_engine import (
    VerificationStatus,
)


@dataclass(frozen=True)
class ResponseOutput:
    """A user-facing response derived from verified orchestration."""

    content: str
    success: bool
    execution_id: str | None
    tool_name: str | None


class ResponseEngine:
    """
    Converts orchestration results into readable user-facing responses.

    Success is reported only when independent verification confirms
    the intended outcome.

    This engine uses deterministic templates and makes no LLM requests.
    """

    MAX_RESULT_LENGTH = 4000

    def generate(
        self,
        orchestration: OrchestrationResult,
        user_request: str | None = None,
    ) -> ResponseOutput:
        """Generate a response from an orchestration result."""

        if not isinstance(orchestration, OrchestrationResult):
            raise TypeError(
                "orchestration must be an OrchestrationResult instance."
            )

        execution = orchestration.execution
        verification = orchestration.verification

        execution_id = (
            execution.execution_id
            if execution is not None
            else None
        )

        tool_name = (
            execution.tool_name
            if execution is not None
            else None
        )

        # Fail closed: every condition must be satisfied before
        # SILVANA can report successful completion.
        verified_success = (
            orchestration.status
            == OrchestrationStatus.VERIFICATION_SUCCEEDED
            and execution is not None
            and execution.success is True
            and verification is not None
            and verification.status == VerificationStatus.VERIFIED
            and verification.verified is True
            and verification.execution_id == execution.execution_id
            and verification.tool_name == execution.tool_name
        )

        if verified_success:
            content = self._success_response(
                execution=execution,
                user_request=user_request,
            )

        elif (
            orchestration.status
            == OrchestrationStatus.EXECUTION_FAILED
        ):
            if execution is not None:
                content = self._failure_response(execution)
            else:
                content = (
                    "I couldn't execute that operation. "
                    "Please try again."
                )

        else:
            content = self._orchestration_failure_response(
                orchestration
            )

        return ResponseOutput(
            content=content,
            success=verified_success,
            execution_id=execution_id,
            tool_name=tool_name,
        )

    def _success_response(
        self,
        execution: ExecutionResult,
        user_request: str | None,
    ) -> str:
        """Create a response only after independent verification."""

        if user_request and user_request.strip():
            opening = (
                f"I've completed and verified your request: "
                f"{user_request.strip()}"
            )
        else:
            tool_label = execution.tool_name.replace("_", " ")
            opening = (
                f"The {tool_label} operation completed "
                "and was independently verified."
            )

        result_text = self._format_result(execution.result)

        if not result_text:
            return opening

        return f"{opening}\n\nVerified result:\n{result_text}"

    def _failure_response(
        self,
        execution: ExecutionResult,
    ) -> str:
        """Translate internal execution errors into safe messages."""

        error = (execution.error or "").lower()

        if "confirmation" in error:
            return (
                "I need your confirmation before I can "
                "perform that action."
            )

        if "permission" in error:
            return (
                "I don't have the required permission "
                "to perform that action."
            )

        if (
            "argument" in error
            or "invalid type" in error
            or "missing required" in error
            or "unexpected arguments" in error
        ):
            return (
                "The request contains invalid or incomplete "
                "information. Please review the details and try again."
            )

        if "not available" in error or "disabled" in error:
            return (
                "That capability is currently unavailable."
            )

        return (
            "I couldn't complete that action. "
            "Please try again."
        )

    def _orchestration_failure_response(
        self,
        orchestration: OrchestrationResult,
    ) -> str:
        """Explain why an operation cannot be reported as successful."""

        status = orchestration.status

        if status == OrchestrationStatus.NO_CAPABILITY:
            return (
                "I don't currently have a tool capable of "
                "performing that request."
            )

        if status == OrchestrationStatus.SELECTION_REQUIRED:
            return (
                "I found multiple possible capabilities, "
                "but need to determine which one is appropriate "
                "before proceeding."
            )

        if status == OrchestrationStatus.INVALID_SELECTION:
            return (
                "The selected capability is not available "
                "for this request. No operation was completed."
            )

        if status == OrchestrationStatus.VERIFICATION_FAILED:
            return (
                "The operation ran, but independent verification "
                "did not confirm the intended outcome. "
                "I cannot report it as successfully completed."
            )

        if status == OrchestrationStatus.VERIFICATION_UNVERIFIABLE:
            return (
                "The operation ran, but I couldn't independently "
                "verify its outcome. I won't claim that it "
                "was completed successfully."
            )

        return (
            "I couldn't establish that the requested operation "
            "was completed successfully."
        )

    def _format_result(self, result: object) -> str:
        """Convert tool output into readable text."""

        if result is None:
            return ""

        if isinstance(result, (dict, list)):
            formatted = json.dumps(
                result,
                indent=2,
                ensure_ascii=False,
                default=str,
            )
        else:
            formatted = str(result)

        if len(formatted) > self.MAX_RESULT_LENGTH:
            formatted = (
                formatted[:self.MAX_RESULT_LENGTH]
                + "\n... (result truncated)"
            )

        return formatted
