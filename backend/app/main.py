
from contextlib import asynccontextmanager

from fastapi import FastAPI

from backend.app.api.router import api_router
from backend.app.core.config import settings

from backend.app.intelligence.provider_manager import ProviderManager
from backend.app.intelligence.tool_registry import ToolRegistry
from backend.app.intelligence.tool_security import ToolSecurityManager
from backend.app.intelligence.capability_resolver import CapabilityResolver
from backend.app.intelligence.execution_engine import ExecutionEngine
from backend.app.intelligence.tool_orchestrator import ToolOrchestrator
from backend.app.intelligence.tool_planner import ToolPlanner
from backend.app.intelligence.verification_engine import VerificationEngine

from backend.app.intelligence.filesystem.workspace import WorkspaceBoundary
from backend.app.intelligence.filesystem.service import FilesystemService
from backend.app.intelligence.filesystem.registration import (
    register_filesystem_tools,
)

from backend.app.intelligence.documents.repository import (
    DocumentRepository,
)
from backend.app.intelligence.documents.storage import (
    DocumentStorage,
)
from backend.app.intelligence.documents.extractor import (
    DocumentExtractor,
)
from backend.app.intelligence.documents.chunker import (
    DocumentChunker,
)
from backend.app.intelligence.documents.indexing import (
    DocumentIndexingService,
)

from backend.app.intelligence.embeddings.fastembed import (
    FastEmbedProvider,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize the LLM provider manager.
    app.state.provider_manager = ProviderManager()

    try:
        # Initialize the tool registry.
        app.state.tool_registry = ToolRegistry()

        # Initialize tool security with read-only permissions.
        app.state.tool_security = ToolSecurityManager(
            granted_permissions={"read"}
        )

        # Initialize independent outcome verification.
        app.state.verification_engine = VerificationEngine()

        # Establish the permitted filesystem workspace.
        app.state.workspace_boundary = WorkspaceBoundary(
            settings.SILVANA_WORKSPACE_ROOT
        )

        # Initialize the bounded filesystem service.
        app.state.filesystem_service = FilesystemService(
            app.state.workspace_boundary
        )

        # Register read-only filesystem capabilities and verifiers.
        register_filesystem_tools(
            registry=app.state.tool_registry,
            verification_engine=app.state.verification_engine,
            service=app.state.filesystem_service,
        )

        # Initialize capability discovery.
        app.state.capability_resolver = CapabilityResolver(
            registry=app.state.tool_registry
        )

        # Initialize deterministic natural-language tool planning.
        app.state.tool_planner = ToolPlanner()

        # Initialize controlled tool execution.
        app.state.execution_engine = ExecutionEngine(
            registry=app.state.tool_registry,
            security_manager=app.state.tool_security,
        )

        # Initialize the orchestration layer.
        app.state.tool_orchestrator = ToolOrchestrator(
            capability_resolver=app.state.capability_resolver,
            execution_engine=app.state.execution_engine,
            verification_engine=app.state.verification_engine,
        )

        # -------------------------------------------------
        # DOCUMENT INTELLIGENCE INFRASTRUCTURE
        # -------------------------------------------------

        # Keep the SQLite database separate from uploaded files.
        document_database_path = (
            settings.SILVANA_DOCUMENT_STORAGE_ROOT.parent
            / "documents.sqlite3"
        )

        app.state.document_database_path = document_database_path

        # Initialize persistent document metadata, chunks,
        # and vector embedding storage.
        app.state.document_repository = DocumentRepository(
            database_path=document_database_path
        )

        app.state.document_repository.initialize()

        # Initialize private document file storage.
        app.state.document_storage = DocumentStorage(
            storage_root=settings.SILVANA_DOCUMENT_STORAGE_ROOT,
            max_file_size_bytes=(
                settings.SILVANA_DOCUMENT_MAX_FILE_SIZE_BYTES
            ),
        )

        # Initialize document text extraction.
        app.state.document_extractor = DocumentExtractor()

        # Construct the local embedding provider.
        # The actual model remains lazy-loaded until inference.
        app.state.embedding_provider = FastEmbedProvider()

        # Initialize token-aware document chunking.
        app.state.document_chunker = DocumentChunker(
            token_counter=app.state.embedding_provider.count_tokens,
            max_tokens=450,
            token_overlap=60,
        )

        # Coordinate extraction, chunking, embeddings,
        # and transactional document indexing.
        app.state.document_indexing_service = DocumentIndexingService(
            repository=app.state.document_repository,
            storage=app.state.document_storage,
            extractor=app.state.document_extractor,
            chunker=app.state.document_chunker,
            embedding_provider=app.state.embedding_provider,
        )

        yield

    finally:
        # Release provider resources on application shutdown.
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
