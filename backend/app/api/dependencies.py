
from fastapi import Request

from backend.app.intelligence.provider_manager import ProviderManager
from backend.app.intelligence.tool_orchestrator import ToolOrchestrator
from backend.app.intelligence.tool_planner import ToolPlanner
from backend.app.intelligence.verification_engine import VerificationEngine

from backend.app.intelligence.documents.repository import (
    DocumentRepository,
)
from backend.app.intelligence.documents.storage import (
    DocumentStorage,
)
from backend.app.intelligence.documents.indexing import (
    DocumentIndexingService,
)


def get_provider_manager(request: Request) -> ProviderManager:
    return request.app.state.provider_manager


def get_tool_orchestrator(request: Request) -> ToolOrchestrator:
    return request.app.state.tool_orchestrator


def get_tool_planner(request: Request) -> ToolPlanner:
    return request.app.state.tool_planner


def get_verification_engine(request: Request) -> VerificationEngine:
    return request.app.state.verification_engine


def get_document_repository(request: Request) -> DocumentRepository:
    return request.app.state.document_repository


def get_document_storage(request: Request) -> DocumentStorage:
    return request.app.state.document_storage


def get_document_indexing_service(
    request: Request,
) -> DocumentIndexingService:
    return request.app.state.document_indexing_service
