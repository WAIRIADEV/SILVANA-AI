
import pytest

from backend.app.intelligence.execution_engine import ExecutionResult
from backend.app.intelligence.response_engine import (
    ResponseEngine,
    ResponseOutput,
)
from backend.app.intelligence.tool_orchestrator import (
    OrchestrationResult,
    OrchestrationStatus,
)
from backend.app.intelligence.verification_engine import (
    VerificationResult,
    VerificationStatus,
)


@pytest.fixture
def engine():
    return ResponseEngine()


def make_execution(
    success=True,
    result=None,
    error=None,
    tool_name="test_tool",
    execution_id="test-execution-001",
):
    return ExecutionResult(
        execution_id=execution_id,
        tool_name=tool_name,
        success=success,
        result=result,
        error=error,
        duration_ms=12.5,
    )


def make_verification(
    status=VerificationStatus.VERIFIED,
    verified=True,
    execution_id="test-execution-001",
    tool_name="test_tool",
    evidence=True,
):
    return VerificationResult(
        execution_id=execution_id,
        tool_name=tool_name,
        status=status,
        verified=verified,
        message="Test verification result.",
        evidence=evidence,
    )


def make_orchestration(
    status=OrchestrationStatus.VERIFICATION_SUCCEEDED,
    execution=None,
    verification=None,
    message="",
):
    return OrchestrationResult(
        status=status,
        available_tools=["test_tool"],
        execution=execution,
        verification=verification,
        message=message,
    )


def make_verified_orchestration(
    result=None,
    tool_name="test_tool",
    execution_id="test-execution-001",
):
    execution = make_execution(
        result=result,
        tool_name=tool_name,
        execution_id=execution_id,
    )

    verification = make_verification(
        tool_name=tool_name,
        execution_id=execution_id,
    )

    return make_orchestration(
        execution=execution,
        verification=verification,
    )


# ---------------------------------------------------------
# VERIFIED SUCCESS RESPONSES
# ---------------------------------------------------------

def test_verified_success_includes_user_request(engine):
    orchestration = make_verified_orchestration(
        result="Projects folder created."
    )

    response = engine.generate(
        orchestration,
        user_request="Create a Projects folder.",
    )

    assert response.success is True
    assert "Create a Projects folder." in response.content
    assert "Projects folder created." in response.content
    assert "verified" in response.content.lower()


def test_verified_success_without_user_request(engine):
    orchestration = make_verified_orchestration(
        result="Operation complete."
    )

    response = engine.generate(orchestration)

    assert response.success is True
    assert "test tool" in response.content
    assert "Operation complete." in response.content


def test_verified_success_handles_empty_result(engine):
    orchestration = make_verified_orchestration(
        result=None
    )

    response = engine.generate(orchestration)

    assert response.success is True
    assert response.content
    assert "verified" in response.content.lower()


def test_dictionary_results_are_formatted(engine):
    orchestration = make_verified_orchestration(
        result={
            "status": "operational",
            "cpu_usage": 25,
        }
    )

    response = engine.generate(orchestration)

    assert '"status": "operational"' in response.content
    assert '"cpu_usage": 25' in response.content


def test_list_results_are_formatted(engine):
    orchestration = make_verified_orchestration(
        result=["report.pdf", "data.csv"]
    )

    response = engine.generate(orchestration)

    assert "report.pdf" in response.content
    assert "data.csv" in response.content


# ---------------------------------------------------------
# VERIFICATION FAILURE RESPONSES
# ---------------------------------------------------------

def test_failed_verification_does_not_claim_success(engine):
    execution = make_execution(
        result="Operation executed."
    )

    verification = make_verification(
        status=VerificationStatus.FAILED,
        verified=False,
        evidence=False,
    )

    orchestration = make_orchestration(
        status=OrchestrationStatus.VERIFICATION_FAILED,
        execution=execution,
        verification=verification,
    )

    response = engine.generate(orchestration)

    assert response.success is False
    assert "did not confirm" in response.content.lower()
    assert "cannot report it as successfully completed" in response.content.lower()


def test_unverifiable_execution_does_not_claim_success(engine):
    execution = make_execution(
        result="Operation executed."
    )

    verification = make_verification(
        status=VerificationStatus.UNVERIFIABLE,
        verified=False,
        evidence=None,
    )

    orchestration = make_orchestration(
        status=OrchestrationStatus.VERIFICATION_UNVERIFIABLE,
        execution=execution,
        verification=verification,
    )

    response = engine.generate(orchestration)

    assert response.success is False
    assert "couldn't independently verify" in response.content.lower()
    assert "won't claim" in response.content.lower()


def test_verified_status_with_false_verified_flag_fails_closed(engine):
    execution = make_execution()

    verification = make_verification(
        status=VerificationStatus.VERIFIED,
        verified=False,
    )

    orchestration = make_orchestration(
        execution=execution,
        verification=verification,
    )

    response = engine.generate(orchestration)

    assert response.success is False
    assert "couldn't establish" in response.content.lower()


def test_mismatched_execution_id_fails_closed(engine):
    execution = make_execution()

    verification = make_verification(
        execution_id="different-execution-002",
    )

    orchestration = make_orchestration(
        execution=execution,
        verification=verification,
    )

    response = engine.generate(orchestration)

    assert response.success is False


def test_mismatched_tool_name_fails_closed(engine):
    execution = make_execution(
        tool_name="test_tool",
    )

    verification = make_verification(
        tool_name="different_tool",
    )

    orchestration = make_orchestration(
        execution=execution,
        verification=verification,
    )

    response = engine.generate(orchestration)

    assert response.success is False


def test_missing_verification_fails_closed(engine):
    execution = make_execution()

    orchestration = make_orchestration(
        execution=execution,
        verification=None,
    )

    response = engine.generate(orchestration)

    assert response.success is False


# ---------------------------------------------------------
# EXECUTION FAILURE RESPONSES
# ---------------------------------------------------------

def test_confirmation_failure_is_explained(engine):
    execution = make_execution(
        success=False,
        error="Explicit user confirmation is required.",
    )

    orchestration = make_orchestration(
        status=OrchestrationStatus.EXECUTION_FAILED,
        execution=execution,
    )

    response = engine.generate(orchestration)

    assert response.success is False
    assert "confirmation" in response.content.lower()


def test_permission_failure_is_explained(engine):
    execution = make_execution(
        success=False,
        error="Permission 'system' has not been granted.",
    )

    orchestration = make_orchestration(
        status=OrchestrationStatus.EXECUTION_FAILED,
        execution=execution,
    )

    response = engine.generate(orchestration)

    assert response.success is False
    assert "permission" in response.content.lower()


def test_invalid_argument_failure_is_explained(engine):
    execution = make_execution(
        success=False,
        error="Invalid type at arguments.value.",
    )

    orchestration = make_orchestration(
        status=OrchestrationStatus.EXECUTION_FAILED,
        execution=execution,
    )

    response = engine.generate(orchestration)

    assert response.success is False
    assert "invalid or incomplete" in response.content.lower()


def test_unavailable_tool_failure_is_explained(engine):
    execution = make_execution(
        success=False,
        error="Requested tool is not available.",
    )

    orchestration = make_orchestration(
        status=OrchestrationStatus.EXECUTION_FAILED,
        execution=execution,
    )

    response = engine.generate(orchestration)

    assert response.success is False
    assert "unavailable" in response.content.lower()


def test_unexpected_failure_does_not_expose_internal_error(engine):
    execution = make_execution(
        success=False,
        error="Database password authentication failed.",
    )

    orchestration = make_orchestration(
        status=OrchestrationStatus.EXECUTION_FAILED,
        execution=execution,
    )

    response = engine.generate(orchestration)

    assert response.success is False
    assert "Database password" not in response.content
    assert "couldn't complete" in response.content.lower()


def test_execution_failure_without_execution_is_handled(engine):
    orchestration = make_orchestration(
        status=OrchestrationStatus.EXECUTION_FAILED,
        execution=None,
    )

    response = engine.generate(orchestration)

    assert response.success is False
    assert "couldn't execute" in response.content.lower()


# ---------------------------------------------------------
# ORCHESTRATION STATUS RESPONSES
# ---------------------------------------------------------

@pytest.mark.parametrize(
    ("status", "expected_text"),
    [
        (
            OrchestrationStatus.NO_CAPABILITY,
            "don't currently have a tool",
        ),
        (
            OrchestrationStatus.SELECTION_REQUIRED,
            "need to determine",
        ),
        (
            OrchestrationStatus.INVALID_SELECTION,
            "not available",
        ),
    ],
)
def test_non_execution_statuses_are_explained(
    engine,
    status,
    expected_text,
):
    orchestration = make_orchestration(
        status=status,
    )

    response = engine.generate(orchestration)

    assert response.success is False
    assert expected_text in response.content.lower()


# ---------------------------------------------------------
# RESPONSE METADATA AND VALIDATION
# ---------------------------------------------------------

def test_execution_metadata_is_preserved(engine):
    orchestration = make_verified_orchestration(
        result="Done.",
        tool_name="create_folder",
    )

    response = engine.generate(orchestration)

    assert isinstance(response, ResponseOutput)
    assert response.execution_id == "test-execution-001"
    assert response.tool_name == "create_folder"


def test_missing_execution_has_no_execution_metadata(engine):
    orchestration = make_orchestration(
        status=OrchestrationStatus.NO_CAPABILITY,
    )

    response = engine.generate(orchestration)

    assert response.execution_id is None
    assert response.tool_name is None


def test_large_results_are_truncated(engine):
    orchestration = make_verified_orchestration(
        result="A" * 5000,
    )

    response = engine.generate(orchestration)

    assert "... (result truncated)" in response.content
    assert len(response.content) < 5000


def test_invalid_orchestration_type_is_rejected(engine):
    with pytest.raises(TypeError):
        engine.generate("not an orchestration result")
