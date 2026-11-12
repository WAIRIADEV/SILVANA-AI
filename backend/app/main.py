
from contextlib import asynccontextmanager

from fastapi import FastAPI

from backend.app.api.router import api_router
from backend.app.intelligence.provider_manager import ProviderManager


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize the provider manager.
    app.state.provider_manager = ProviderManager()

    yield

    # Release initialized provider resources on shutdown.
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
