
import inspect
import logging
import time
import uuid


from dataclasses import dataclass
from enum import Enum
from typing import Any, Awaitable, Callable

from backend.app.intelligence.tool_registry import ToolRegistry
from backend.app.intelligence.tool_security import (
    ToolArgumentError,
    ToolPermissionError,
    ToolSecurityManager,
)


logger = logging.getLogger(__name__)

StateObserver = Callable[
    [str],
    Awaitable[None] | None,
]


class ExecutionStatus(str, Enum):
    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ExecutionResult:
    """
    Structured result returned after a tool execution attempt.
    """

    execution_id: str
    tool_name: str
    success: bool

    result: Any = None
    error: str | None = None
    duration_ms: float = 0.0


class ExecutionEngine:
    """
    Coordinates safe execution of registered SILVANA tools.

    Execution sequence:
    1. Retrieve the registered tool.
    2. Check authorization.
    3. Validate arguments.
    4. Notify observers that execution is beginning.
    5. Invoke the registered handler.
    6. Return a structured execution result.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        security_manager: ToolSecurityManager,
    ):
        self.registry = registry
        self.security_manager = security_manager

    async def _notify_state(
        self,
        observer: StateObserver | None,
        state: str,
    ) -> None:
        """
        Notify an optional observer without allowing monitoring
        failures to interfere with tool execution.
        """

        if observer is None:
            return

        try:
            result = observer(state)

            if inspect.isawaitable(result):
                await result

        except Exception:
            logger.exception(
                "Execution state observer failed. State: %s",
                state,
            )

    async def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        user_confirmed: bool = False,
        on_state_change: StateObserver | None = None,
    ) -> ExecutionResult:

        execution_id = str(uuid.uuid4())
        start_time = time.perf_counter()

        def create_result(
            success: bool,
            result: Any = None,
            error: str | None = None,
        ) -> ExecutionResult:

            duration_ms = (
                time.perf_counter() - start_time
            ) * 1000

            return ExecutionResult(
                execution_id=execution_id,
                tool_name=tool_name,
                success=success,
                result=result,
                error=error,
                duration_ms=round(duration_ms, 2),
            )

        # Step 1: Retrieve the registered tool.
        try:
            tool = self.registry.get(tool_name)

        except KeyError:
            logger.warning(
                "Execution rejected: tool unavailable. ID: %s",
                execution_id,
            )

            return create_result(
                success=False,
                error="Requested tool is not available.",
            )

        # Step 2: Check authorization and confirmation.
        try:
            self.security_manager.authorize(
                tool=tool,
                user_confirmed=user_confirmed,
            )

        except ToolPermissionError as error:
            logger.warning(
                "Execution denied by permission policy. "
                "Tool: %s | ID: %s",
                tool_name,
                execution_id,
            )

            return create_result(
                success=False,
                error=str(error),
            )

        # Step 3: Validate the supplied arguments.
        try:
            validated_arguments = (
                self.security_manager.validate_arguments(
                    tool=tool,
                    arguments=arguments,
                )
            )

        except ToolArgumentError as error:
            logger.warning(
                "Execution rejected due to invalid arguments. "
                "Tool: %s | ID: %s",
                tool_name,
                execution_id,
            )

            return create_result(
                success=False,
                error=str(error),
            )

        # Step 4: Notify observers only after all security checks.
        await self._notify_state(
            on_state_change,
            "executing",
        )

        # Step 5: Execute the registered handler.
        try:
            handler_result = tool.handler(
                **validated_arguments
            )

            if inspect.isawaitable(handler_result):
                handler_result = await handler_result

        except Exception:
            logger.exception(
                "Tool execution failed. Tool: %s | ID: %s",
                tool_name,
                execution_id,
            )

            return create_result(
                success=False,
                error="Tool execution failed.",
            )

        # Step 6: Return the successful result.
        logger.info(
            "Tool execution completed. "
            "Tool: %s | ID: %s",
            tool_name,
            execution_id,
        )

        return create_result(
            success=True,
            result=handler_result,
        )
