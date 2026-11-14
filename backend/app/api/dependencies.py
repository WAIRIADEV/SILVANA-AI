from fastapi import Request

from backend.app.intelligence.provider_manager import ProviderManager
from backend.app.intelligence.tool_orchestrator import ToolOrchestrator
from backend.app.intelligence.verification_engine import VerificationEngine


def get_provider_manager(request: Request) -> ProviderManager:
    return request.app.state.provider_manager


def get_tool_orchestrator(request: Request) -> ToolOrchestrator:
    return request.app.state.tool_orchestrator


def get_verification_engine(request: Request) -> VerificationEngine:
    return request.app.state.verification_engine
