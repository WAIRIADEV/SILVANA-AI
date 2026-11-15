
import pytest

from backend.app.intelligence.filesystem.workspace import WorkspaceBoundary
from backend.app.intelligence.filesystem.service import FilesystemService
from backend.app.intelligence.filesystem.registration import (
    register_filesystem_tools,
)
from backend.app.intelligence.tool_registry import ToolRegistry
from backend.app.intelligence.tool_security import (
    ToolSecurityManager,
    ToolArgumentError,
)
from backend.app.intelligence.verification_engine import (
    VerificationEngine,
    VerificationStatus,
)


@pytest.fixture
def setup_filesystem(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()

    (root / "Documents").mkdir()
    (root / "Documents" / "report.txt").write_text(
        "Test report",
        encoding="utf-8",
    )

    boundary = WorkspaceBoundary(root)
    service = FilesystemService(boundary)

    registry = ToolRegistry()
    verification = VerificationEngine()

    register_filesystem_tools(
        registry,
        verification,
        service,
    )

    return registry, verification, service


def test_registers_all_four_filesystem_tools(setup_filesystem):
    registry, _, _ = setup_filesystem

    names = {tool.name for tool in registry.list_tools()}

    assert names == {
        "list_directory",
        "search_files",
        "get_file_metadata",
        "check_file_exists",
    }


def test_all_filesystem_tools_are_read_only(setup_filesystem):
    registry, _, _ = setup_filesystem

    for tool in registry.list_tools():
        assert tool.permission_level == "read"
        assert tool.requires_confirmation is False


def test_all_filesystem_tools_support_tool_action(setup_filesystem):
    registry, _, _ = setup_filesystem

    for tool in registry.list_tools():
        assert tool.supported_intents == {"tool_action"}


def test_all_tools_have_strict_argument_schemas(setup_filesystem):
    registry, _, _ = setup_filesystem

    for tool in registry.list_tools():
        assert tool.parameters["type"] == "object"
        assert tool.parameters["additionalProperties"] is False


def test_rejects_unexpected_arguments(setup_filesystem):
    registry, _, _ = setup_filesystem
    security = ToolSecurityManager()

    tool = registry.get("check_file_exists")

    with pytest.raises(ToolArgumentError):
        security.validate_arguments(
            tool,
            {
                "path": "Documents/report.txt",
                "unexpected": True,
            },
        )


def test_rejects_missing_required_path(setup_filesystem):
    registry, _, _ = setup_filesystem
    security = ToolSecurityManager()

    tool = registry.get("get_file_metadata")

    with pytest.raises(ToolArgumentError):
        security.validate_arguments(tool, {})


def test_rejects_excessive_search_depth(setup_filesystem):
    registry, _, _ = setup_filesystem
    security = ToolSecurityManager()

    tool = registry.get("search_files")

    with pytest.raises(ToolArgumentError):
        security.validate_arguments(
            tool,
            {"max_depth": 11},
        )


def test_registers_verifiers_for_all_tools(setup_filesystem):
    _, verification, _ = setup_filesystem

    assert len(verification._verifiers) == 4


@pytest.mark.parametrize(
    "tool_name,arguments",
    [
        ("list_directory", {"path": "Documents"}),
        ("search_files", {"pattern": "*.txt"}),
        ("get_file_metadata", {"path": "Documents/report.txt"}),
        ("check_file_exists", {"path": "Documents/report.txt"}),
    ],
)
def test_each_tool_produces_independently_verified_result(
    setup_filesystem,
    tool_name,
    arguments,
):
    registry, verification, _ = setup_filesystem

    from backend.app.intelligence.execution_engine import ExecutionEngine

    execution_engine = ExecutionEngine(
        registry=registry,
        security_manager=ToolSecurityManager(),
    )

    import asyncio

    execution = asyncio.run(
        execution_engine.execute(
            tool_name=tool_name,
            arguments=arguments,
        )
    )

    assert execution.success is True

    result = verification.verify(
        execution_id=execution.execution_id,
        tool_name=tool_name,
        execution_success=execution.success,
        execution_result=execution.result,
    )

    assert result.status == VerificationStatus.VERIFIED
    assert result.verified is True


def test_directory_verification_rejects_fabricated_entry(setup_filesystem):
    _, verification, _ = setup_filesystem

    result = {
        "operation": "list_directory",
        "path": "Documents",
        "entries": [
            {"name": "fake.txt", "kind": "file"},
        ],
        "count": 1,
        "truncated": False,
    }

    verification_result = verification.verify(
        execution_id="test-123",
        tool_name="list_directory",
        execution_success=True,
        execution_result=result,
    )

    assert verification_result.status == VerificationStatus.FAILED
    assert verification_result.verified is False


def test_metadata_verification_rejects_fabricated_size(setup_filesystem):
    _, verification, service = setup_filesystem

    result = service.get_file_metadata("Documents/report.txt")
    result["size_bytes"] = 999999

    verification_result = verification.verify(
        execution_id="test-456",
        tool_name="get_file_metadata",
        execution_success=True,
        execution_result=result,
    )

    assert verification_result.status == VerificationStatus.FAILED


def test_existence_verification_rejects_fabricated_result(setup_filesystem):
    _, verification, _ = setup_filesystem

    result = {
        "operation": "check_file_exists",
        "path": "missing.txt",
        "exists": True,
        "kind": "file",
    }

    verification_result = verification.verify(
        execution_id="test-789",
        tool_name="check_file_exists",
        execution_success=True,
        execution_result=result,
    )

    assert verification_result.status == VerificationStatus.FAILED
