
from contextlib import asynccontextmanager

from fastapi import FastAPI

from backend.app.api.router import api_router
from backend.app.intelligence.provider_manager import ProviderManager
from backend.app.intelligence.tool_registry import ToolRegistry
from backend.app.intelligence.tool_security import ToolSecurityManager
from backend.app.intelligence.capability_resolver import CapabilityResolver
from backend.app.intelligence.execution_engine import ExecutionEngine
from backend.app.intelligence.tool_orchestrator import ToolOrchestrator
from backend.app.intelligence.verification_engine import VerificationEngine


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize the LLM provider manager.
    app.state.provider_manager = ProviderManager()

    # Initialize the tool registry.
    app.state.tool_registry = ToolRegistry()

    # Initialize tool security with read-only permissions.
    app.state.tool_security = ToolSecurityManager(
        granted_permissions={"read"}
    )

    # Initialize capability discovery.
    app.state.capability_resolver = CapabilityResolver(
        registry=app.state.tool_registry
    )

    # Initialize controlled tool execution.
    app.state.execution_engine = ExecutionEngine(
        registry=app.state.tool_registry,
        security_manager=app.state.tool_security,
    )

    # Initialize independent outcome verification.
    app.state.verification_engine = VerificationEngine()

    # Initialize the orchestration layer.
    app.state.tool_orchestrator = ToolOrchestrator(
        capability_resolver=app.state.capability_resolver,
        execution_engine=app.state.execution_engine,
        verification_engine=app.state.verification_engine,
    )

    try:
        yield
    finally:
        # Release provider resources on shutdown.
        await app.state.provider_manager.aclose()


app = FastAPI(
    title="SILVANA",
    description="AI-powered intelligent assistant.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/")
async def root():
    return {
        "application": "SILVANA",
        "message": "Observe. Understand. Execute. Verify.",
        "documentation": "/docs",
    }


app.include_router(api_router, prefix="/api/v1")
