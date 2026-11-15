
import asyncio

import pytest

from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.intelligence.intent_engine import Intent, IntentResult
from backend.app.intelligence.tool_orchestrator import OrchestrationStatus
from backend.app.intelligence.verification_engine import VerificationStatus


class FakeProviderManager:
    async def aclose(self):
        pass


@pytest.fixture
def filesystem_app(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "backend.app.main.ProviderManager",
        FakeProviderManager,
    )

    monkeypatch.setattr(
        "backend.app.main.settings.SILVANA_WORKSPACE_ROOT",
        tmp_path,
    )

    (tmp_path / "notes.txt").write_text(
        "SILVANA filesystem integration test.",
        encoding="utf-8",
    )

    (tmp_path / "reports").mkdir()

    with TestClient(app):
        yield app


def make_intent():
    return IntentResult(
        intent=Intent.TOOL_ACTION,
        confidence=0.95,
        requires_tool=True,
        requires_confirmation=False,
        matched_rule="explicit_action",
    )


@pytest.mark.parametrize(
    "tool_name, arguments",
    [
        ("list_directory", {"path": "."}),
        (
            "search_files",
            {"directory": ".", "pattern": "*.txt"},
        ),
        ("get_file_metadata", {"path": "notes.txt"}),
        ("check_file_exists", {"path": "notes.txt"}),
    ],
)
def test_filesystem_tools_execute_and_verify(
    filesystem_app,
    tool_name,
    arguments,
):
    orchestrator = filesystem_app.state.tool_orchestrator

    result = asyncio.run(
        orchestrator.orchestrate(
            intent=make_intent(),
            selected_tool=tool_name,
            arguments=arguments,
        )
    )

    assert result.status == OrchestrationStatus.VERIFICATION_SUCCEEDED

    assert result.execution is not None
    assert result.execution.success is True

    assert result.verification is not None
    assert result.verification.status == VerificationStatus.VERIFIED
    assert result.verification.verified is True


def test_filesystem_discovery_requires_explicit_selection(
    filesystem_app,
):
    orchestrator = filesystem_app.state.tool_orchestrator

    result = asyncio.run(
        orchestrator.orchestrate(
            intent=make_intent(),
        )
    )

    assert result.status == OrchestrationStatus.SELECTION_REQUIRED

    assert set(result.available_tools) == {
        "list_directory",
        "search_files",
        "get_file_metadata",
        "check_file_exists",
    }

    assert result.execution is None
    assert result.verification is None
