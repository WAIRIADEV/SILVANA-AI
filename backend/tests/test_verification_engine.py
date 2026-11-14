
import pytest

from backend.app.intelligence.verification_engine import (
    VerificationEngine,
    VerificationStatus,
)


@pytest.fixture
def engine():
    return VerificationEngine()


def test_successful_outcome_is_verified(engine):
    engine.register(
        "create_folder",
        lambda result: result["exists"] is True,
    )

    result = engine.verify(
        execution_id="execution-001",
        tool_name="create_folder",
        execution_success=True,
        execution_result={"exists": True},
    )

    assert result.status == VerificationStatus.VERIFIED
    assert result.verified is True
    assert result.execution_id == "execution-001"


def test_unsuccessful_outcome_fails_verification(engine):
    engine.register(
        "create_folder",
        lambda result: result["exists"] is True,
    )

    result = engine.verify(
        execution_id="execution-002",
        tool_name="create_folder",
        execution_success=True,
        execution_result={"exists": False},
    )

    assert result.status == VerificationStatus.FAILED
    assert result.verified is False


def test_missing_verifier_returns_unverifiable(engine):
    result = engine.verify(
        execution_id="execution-003",
        tool_name="unknown_tool",
        execution_success=True,
        execution_result="completed",
    )

    assert result.status == VerificationStatus.UNVERIFIABLE
    assert result.verified is False


def test_failed_execution_cannot_be_verified(engine):
    engine.register(
        "create_folder",
        lambda result: True,
    )

    result = engine.verify(
        execution_id="execution-004",
        tool_name="create_folder",
        execution_success=False,
    )

    assert result.status == VerificationStatus.FAILED
    assert result.verified is False


def test_verifier_exception_returns_unverifiable(engine):
    def failing_verifier(result):
        raise RuntimeError("Internal verification error")

    engine.register("check_file", failing_verifier)

    result = engine.verify(
        execution_id="execution-005",
        tool_name="check_file",
        execution_success=True,
    )

    assert result.status == VerificationStatus.UNVERIFIABLE
    assert result.verified is False
    assert "Internal verification error" not in result.message


def test_duplicate_verifier_registration_is_rejected(engine):
    verifier = lambda result: True

    engine.register("check_file", verifier)

    with pytest.raises(ValueError, match="already registered"):
        engine.register("check_file", verifier)


def test_invalid_verifier_is_rejected(engine):
    with pytest.raises(ValueError, match="callable"):
        engine.register("check_file", "not_callable")


def test_verifier_must_return_boolean(engine):
    engine.register(
        "check_file",
        lambda result: "yes",
    )

    result = engine.verify(
        execution_id="execution-006",
        tool_name="check_file",
        execution_success=True,
    )

    assert result.status == VerificationStatus.UNVERIFIABLE
    assert result.verified is False


def test_verifier_can_be_unregistered(engine):
    engine.register(
        "check_file",
        lambda result: True,
    )

    engine.unregister("check_file")

    result = engine.verify(
        execution_id="execution-007",
        tool_name="check_file",
        execution_success=True,
    )

    assert result.status == VerificationStatus.UNVERIFIABLE


def test_unregister_unknown_verifier_is_rejected(engine):
    with pytest.raises(KeyError, match="No verifier"):
        engine.unregister("unknown_tool")


@pytest.mark.parametrize(
    ("execution_id", "tool_name"),
    [
        ("", "check_file"),
        ("execution-008", ""),
    ],
)
def test_verification_rejects_empty_identifiers(
    engine,
    execution_id,
    tool_name,
):
    with pytest.raises(ValueError):
        engine.verify(
            execution_id=execution_id,
            tool_name=tool_name,
            execution_success=True,
        )
