
import pytest

from backend.app.intelligence.execution_engine import ExecutionResult
from backend.app.intelligence.response_engine import (
    ResponseEngine,
    ResponseOutput,
)


@pytest.fixture
def engine():
    return ResponseEngine()


def make_execution(
    success=True,
    result=None,
    error=None,
    tool_name="test_tool",
):
    return ExecutionResult(
        execution_id="test-execution-001",
        tool_name=tool_name,
        success=success,
        result=result,
        error=error,
        duration_ms=12.5,
    )


def test_success_response_includes_user_request(engine):
    execution = make_execution(
        result="Projects folder created."
    )

    response = engine.generate(
        execution,
        user_request="Create a Projects folder.",
    )

    assert response.success is True
    assert "Create a Projects folder." in response.content
    assert "Projects folder created." in response.content


def test_success_response_without_user_request(engine):
    execution = make_execution(
        result="Operation complete."
    )

    response = engine.generate(execution)

    assert response.success is True
    assert "test tool" in response.content
    assert "Operation complete." in response.content


def test_success_response_handles_empty_result(engine):
    execution = make_execution(result=None)

    response = engine.generate(execution)

    assert response.success is True
    assert response.content
    assert "completed successfully" in response.content


def test_dictionary_results_are_formatted(engine):
    execution = make_execution(
        result={
            "status": "operational",
            "cpu_usage": 25,
        }
    )

    response = engine.generate(execution)

    assert '"status": "operational"' in response.content
    assert '"cpu_usage": 25' in response.content


def test_list_results_are_formatted(engine):
    execution = make_execution(
        result=["report.pdf", "data.csv"]
    )

    response = engine.generate(execution)

    assert "report.pdf" in response.content
    assert "data.csv" in response.content


def test_confirmation_failure_is_explained(engine):
    execution = make_execution(
        success=False,
        error="Explicit user confirmation is required.",
    )

    response = engine.generate(execution)

    assert response.success is False
    assert "confirmation" in response.content.lower()


def test_permission_failure_is_explained(engine):
    execution = make_execution(
        success=False,
        error="Permission 'system' has not been granted.",
    )

    response = engine.generate(execution)

    assert response.success is False
    assert "permission" in response.content.lower()


def test_invalid_argument_failure_is_explained(engine):
    execution = make_execution(
        success=False,
        error="Invalid type at arguments.value.",
    )

    response = engine.generate(execution)

    assert response.success is False
    assert "invalid or incomplete" in response.content.lower()


def test_unavailable_tool_failure_is_explained(engine):
    execution = make_execution(
        success=False,
        error="Requested tool is not available.",
    )

    response = engine.generate(execution)

    assert response.success is False
    assert "unavailable" in response.content.lower()


def test_unexpected_failure_does_not_expose_internal_error(engine):
    execution = make_execution(
        success=False,
        error="Database password authentication failed.",
    )

    response = engine.generate(execution)

    assert response.success is False
    assert "Database password" not in response.content
    assert "couldn't complete" in response.content.lower()


def test_execution_metadata_is_preserved(engine):
    execution = make_execution(
        result="Done.",
        tool_name="create_folder",
    )

    response = engine.generate(execution)

    assert isinstance(response, ResponseOutput)
    assert response.execution_id == execution.execution_id
    assert response.tool_name == "create_folder"


def test_large_results_are_truncated(engine):
    execution = make_execution(
        result="A" * 5000,
    )

    response = engine.generate(execution)

    assert "... (result truncated)" in response.content
    assert len(response.content) < 5000


def test_invalid_execution_type_is_rejected(engine):
    with pytest.raises(TypeError):
        engine.generate("not an execution result")
