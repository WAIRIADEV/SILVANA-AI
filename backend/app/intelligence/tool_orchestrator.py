
from dataclasses import dataclass
from enum import Enum
from typing import Any

from backend.app.intelligence.capability_resolver import CapabilityResolver
from backend.app.intelligence.execution_engine import (
    ExecutionEngine,
    ExecutionResult,
)
from backend.app.intelligence.intent_engine import IntentResult
from backend.app.intelligence.verification_engine import (
    VerificationEngine,
    VerificationResult,
    VerificationStatus,
)


class OrchestrationStatus(str, Enum):
    NO_CAPABILITY = "no_capability"
    SELECTION_REQUIRED = "selection_required"
    INVALID_SELECTION = "invalid_selection"
    EXECUTION_FAILED = "execution_failed"
    VERIFICATION_SUCCEEDED = "verification_succeeded"
    VERIFICATION_FAILED = "verification_failed"
    VERIFICATION_UNVERIFIABLE = "verification_unverifiable"


@dataclass
class OrchestrationResult:
    status: OrchestrationStatus
    available_tools: list[str]
    execution: ExecutionResult | None = None
    verification: VerificationResult | None = None
    message: str = ""


class ToolOrchestrator:
    """
    Coordinates capability discovery, controlled execution,
    and independent outcome verification.

    Discovery never executes tools.
    Execution delegates to ExecutionEngine.
    Verification delegates to VerificationEngine.
    """

    def __init__(
        self,
        capability_resolver: CapabilityResolver,
        execution_engine: ExecutionEngine,
        verification_engine: VerificationEngine,
    ):
        self.capability_resolver = capability_resolver
        self.execution_engine = execution_engine
        self.verification_engine = verification_engine

    async def orchestrate(
        self,
        intent: IntentResult,
        arguments: dict[str, Any] | None = None,
        selected_tool: str | None = None,
        user_confirmed: bool = False,
    ) -> OrchestrationResult:

        # Step 1: Discover available capabilities.
        capabilities = self.capability_resolver.resolve(intent)

        available_tools = [
            capability.tool_name
            for capability in capabilities
        ]

        if not capabilities:
            return OrchestrationResult(
                status=OrchestrationStatus.NO_CAPABILITY,
                available_tools=[],
                message="No available tool supports this request.",
            )

        # Step 2: Require explicit tool selection.
        if selected_tool is None:
            return OrchestrationResult(
                status=OrchestrationStatus.SELECTION_REQUIRED,
                available_tools=available_tools,
                message="A tool must be selected before execution.",
            )

        # Step 3: Validate the selected capability.
        if selected_tool not in available_tools:
            return OrchestrationResult(
                status=OrchestrationStatus.INVALID_SELECTION,
                available_tools=available_tools,
                message="The selected tool does not support this request.",
            )

        # Step 4: Execute through the security-controlled engine.
        execution = await self.execution_engine.execute(
            tool_name=selected_tool,
            arguments=arguments if arguments is not None else {},
            user_confirmed=user_confirmed,
        )

        # Step 5: A failed execution cannot be reported as verified.
        if not execution.success:
            verification = self.verification_engine.verify(
                execution_id=execution.execution_id,
                tool_name=execution.tool_name,
                execution_success=False,
                execution_result=execution.result,
            )

            return OrchestrationResult(
                status=OrchestrationStatus.EXECUTION_FAILED,
                available_tools=available_tools,
                execution=execution,
                verification=verification,
                message="Tool execution failed; the intended outcome was not verified.",
            )

        # Step 6: Independently verify the outcome.
        verification = self.verification_engine.verify(
            execution_id=execution.execution_id,
            tool_name=execution.tool_name,
            execution_success=True,
            execution_result=execution.result,
        )

        if verification.status == VerificationStatus.VERIFIED:
            return OrchestrationResult(
                status=OrchestrationStatus.VERIFICATION_SUCCEEDED,
                available_tools=available_tools,
                execution=execution,
                verification=verification,
                message="Tool execution succeeded and its outcome was verified.",
            )

        if verification.status == VerificationStatus.FAILED:
            return OrchestrationResult(
                status=OrchestrationStatus.VERIFICATION_FAILED,
                available_tools=available_tools,
                execution=execution,
                verification=verification,
                message="Tool execution succeeded, but the intended outcome could not be confirmed.",
            )

        return OrchestrationResult(
            status=OrchestrationStatus.VERIFICATION_UNVERIFIABLE,
            available_tools=available_tools,
            execution=execution,
            verification=verification,
            message="Tool execution succeeded, but its outcome could not be independently verified.",
        )
