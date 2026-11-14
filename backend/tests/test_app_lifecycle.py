
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.api.dependencies import (
    get_provider_manager,
    get_tool_orchestrator,
    get_verification_engine,
)
from backend.app.intelligence.capability_resolver import CapabilityResolver
from backend.app.intelligence.execution_engine import ExecutionEngine
from backend.app.intelligence.provider_manager import ProviderManager
from backend.app.intelligence.tool_orchestrator import ToolOrchestrator
from backend.app.intelligence.tool_registry import ToolRegistry
from backend.app.intelligence.tool_security import ToolSecurityManager
from backend.app.intelligence.verification_engine import VerificationEngine


class FakeProviderManager:
    def __init__(self):
        self.closed = False

    async def aclose(self):
        self.closed = True


def test_application_initializes_intelligence_components(monkeypatch):
    fake_manager = FakeProviderManager()

    monkeypatch.setattr(
        "backend.app.main.ProviderManager",
        lambda: fake_manager,
    )

    with TestClient(app):
        assert app.state.provider_manager is fake_manager

        assert isinstance(app.state.tool_registry, ToolRegistry)
        assert isinstance(app.state.tool_security, ToolSecurityManager)
        assert isinstance(app.state.capability_resolver, CapabilityResolver)
        assert isinstance(app.state.execution_engine, ExecutionEngine)
        assert isinstance(app.state.verification_engine, VerificationEngine)
        assert isinstance(app.state.tool_orchestrator, ToolOrchestrator)

        assert app.state.tool_security.granted_permissions == {"read"}


def test_intelligence_components_share_correct_dependencies(monkeypatch):
    fake_manager = FakeProviderManager()

    monkeypatch.setattr(
        "backend.app.main.ProviderManager",
        lambda: fake_manager,
    )

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


def test_dependencies_return_application_shared_instances(monkeypatch):
    fake_manager = FakeProviderManager()

    monkeypatch.setattr(
        "backend.app.main.ProviderManager",
        lambda: fake_manager,
    )

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


def test_application_closes_provider_manager_on_shutdown(monkeypatch):
    fake_manager = FakeProviderManager()

    monkeypatch.setattr(
        "backend.app.main.ProviderManager",
        lambda: fake_manager,
    )

    with TestClient(app):
        assert fake_manager.closed is False

    assert fake_manager.closed is True


def test_application_initializes_verification_engine(monkeypatch):
    fake_manager = FakeProviderManager()

    monkeypatch.setattr(
        "backend.app.main.ProviderManager",
        lambda: fake_manager,
    )

    with TestClient(app):
        assert isinstance(
            app.state.verification_engine,
            VerificationEngine,
        )


def test_verification_dependency_returns_shared_instance(monkeypatch):
    fake_manager = FakeProviderManager()

    monkeypatch.setattr(
        "backend.app.main.ProviderManager",
        lambda: fake_manager,
    )

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
