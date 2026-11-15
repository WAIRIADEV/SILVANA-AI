
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.core.config import settings
from backend.app.api.dependencies import (
    get_provider_manager,
    get_tool_orchestrator,
    get_tool_planner,
    get_verification_engine,
)
from backend.app.intelligence.capability_resolver import CapabilityResolver
from backend.app.intelligence.execution_engine import ExecutionEngine
from backend.app.intelligence.tool_orchestrator import ToolOrchestrator
from backend.app.intelligence.tool_planner import ToolPlanner
from backend.app.intelligence.tool_registry import ToolRegistry
from backend.app.intelligence.tool_security import ToolSecurityManager
from backend.app.intelligence.verification_engine import VerificationEngine
from backend.app.intelligence.filesystem.workspace import WorkspaceBoundary
from backend.app.intelligence.filesystem.service import FilesystemService


class FakeProviderManager:
    def __init__(self):
        self.closed = False

    async def aclose(self):
        self.closed = True


def create_fake_provider(monkeypatch):
    fake_manager = FakeProviderManager()

    monkeypatch.setattr(
        "backend.app.main.ProviderManager",
        lambda: fake_manager,
    )

    return fake_manager


def test_application_initializes_intelligence_components(monkeypatch):
    fake_manager = create_fake_provider(monkeypatch)

    with TestClient(app):
        assert app.state.provider_manager is fake_manager

        assert isinstance(app.state.tool_registry, ToolRegistry)
        assert isinstance(app.state.tool_security, ToolSecurityManager)
        assert isinstance(app.state.capability_resolver, CapabilityResolver)
        assert isinstance(app.state.execution_engine, ExecutionEngine)
        assert isinstance(app.state.verification_engine, VerificationEngine)
        assert isinstance(app.state.tool_orchestrator, ToolOrchestrator)
        assert isinstance(app.state.tool_planner, ToolPlanner)

        assert app.state.tool_security.granted_permissions == {"read"}


def test_intelligence_components_share_correct_dependencies(monkeypatch):
    create_fake_provider(monkeypatch)

    with TestClient(app):
        registry = app.state.tool_registry
        security = app.state.tool_security
        resolver = app.state.capability_resolver
        execution_engine = app.state.execution_engine
        verification_engine = app.state.verification_engine
        orchestrator = app.state.tool_orchestrator

        assert resolver.registry is registry

        assert execution_engine.registry is registry
        assert execution_engine.security_manager is security

        assert orchestrator.capability_resolver is resolver
        assert orchestrator.execution_engine is execution_engine
        assert orchestrator.verification_engine is verification_engine

        assert isinstance(app.state.tool_planner, ToolPlanner)


def test_dependencies_return_application_shared_instances(monkeypatch):
    create_fake_provider(monkeypatch)

    with TestClient(app):
        request = type(
            "FakeRequest",
            (),
            {"app": app},
        )()

        assert get_provider_manager(request) is app.state.provider_manager

        assert (
            get_tool_orchestrator(request)
            is app.state.tool_orchestrator
        )

        assert (
            get_verification_engine(request)
            is app.state.verification_engine
        )

        assert (
            get_tool_planner(request)
            is app.state.tool_planner
        )


def test_application_closes_provider_manager_on_shutdown(monkeypatch):
    fake_manager = create_fake_provider(monkeypatch)

    with TestClient(app):
        assert fake_manager.closed is False

    assert fake_manager.closed is True


def test_application_initializes_verification_engine(monkeypatch):
    create_fake_provider(monkeypatch)

    with TestClient(app):
        assert isinstance(
            app.state.verification_engine,
            VerificationEngine,
        )


def test_verification_dependency_returns_shared_instance(monkeypatch):
    create_fake_provider(monkeypatch)

    with TestClient(app):
        request = type(
            "FakeRequest",
            (),
            {"app": app},
        )()

        assert (
            get_verification_engine(request)
            is app.state.verification_engine
        )


# FILESYSTEM STARTUP INTEGRATION


def test_application_initializes_filesystem_components(monkeypatch):
    create_fake_provider(monkeypatch)

    with TestClient(app):
        assert isinstance(
            app.state.workspace_boundary,
            WorkspaceBoundary,
        )

        assert isinstance(
            app.state.filesystem_service,
            FilesystemService,
        )

        assert (
            app.state.filesystem_service.boundary
            is app.state.workspace_boundary
        )

        assert (
            app.state.workspace_boundary.root
            == settings.SILVANA_WORKSPACE_ROOT.resolve()
        )


def test_application_registers_all_filesystem_tools(monkeypatch):
    create_fake_provider(monkeypatch)

    expected_tools = {
        "list_directory",
        "search_files",
        "get_file_metadata",
        "check_file_exists",
    }

    with TestClient(app):
        registered_tools = {
            tool.name
            for tool in app.state.tool_registry.list_tools()
        }

        assert registered_tools == expected_tools


def test_filesystem_tools_have_read_only_permissions(monkeypatch):
    create_fake_provider(monkeypatch)

    with TestClient(app):
        tools = app.state.tool_registry.list_tools()

        for tool in tools:
            assert tool.permission_level == "read"
            assert tool.requires_confirmation is False
            assert tool.supported_intents == {"tool_action"}


def test_application_registers_filesystem_verifiers(monkeypatch):
    create_fake_provider(monkeypatch)

    expected_verifiers = {
        "list_directory",
        "search_files",
        "get_file_metadata",
        "check_file_exists",
    }

    with TestClient(app):
        registered_verifiers = set(
            app.state.verification_engine._verifiers
        )

        assert registered_verifiers == expected_verifiers


def test_tool_planner_dependency_returns_shared_instance(monkeypatch):
    create_fake_provider(monkeypatch)

    with TestClient(app):
        request = type(
            "FakeRequest",
            (),
            {"app": app},
        )()

        assert (
            get_tool_planner(request)
            is app.state.tool_planner
        )
