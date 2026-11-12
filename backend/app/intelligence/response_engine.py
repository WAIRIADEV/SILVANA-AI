
import json

from dataclasses import dataclass

from backend.app.intelligence.execution_engine import ExecutionResult


@dataclass(frozen=True)
class ResponseOutput:
    """A user-facing response derived from a tool execution."""

    content: str
    success: bool
    execution_id: str
    tool_name: str


class ResponseEngine:
    """
    Converts structured execution results into readable responses.

    This initial implementation uses deterministic templates.
    It does not make additional LLM requests.
    """

    MAX_RESULT_LENGTH = 4000

    def generate(
        self,
        execution: ExecutionResult,
        user_request: str | None = None,
    ) -> ResponseOutput:
        """Generate a user-facing response from an execution result."""

        if not isinstance(execution, ExecutionResult):
            raise TypeError(
                "execution must be an ExecutionResult instance."
            )

        if execution.success:
            content = self._success_response(
                execution=execution,
                user_request=user_request,
            )
        else:
            content = self._failure_response(
                execution=execution,
            )

        return ResponseOutput(
            content=content,
            success=execution.success,
            execution_id=execution.execution_id,
            tool_name=execution.tool_name,
        )

    def _success_response(
        self,
        execution: ExecutionResult,
        user_request: str | None,
    ) -> str:
        """Create a clear response for successful execution."""

        if user_request and user_request.strip():
            opening = (
                f"I've completed your request: "
                f"{user_request.strip()}"
            )
        else:
            tool_label = execution.tool_name.replace("_", " ")
            opening = (
                f"The {tool_label} operation completed successfully."
            )

        result_text = self._format_result(execution.result)

        if not result_text:
            return opening

        return f"{opening}\n\nResult:\n{result_text}"

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
