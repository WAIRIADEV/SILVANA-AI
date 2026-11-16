
from fastapi import APIRouter

from backend.app.api.routes import (
    chat,
    documents,
    health,
    providers,
)

api_router = APIRouter()

api_router.include_router(health.router, tags=["System"])
api_router.include_router(providers.router, tags=["Intelligence"])
api_router.include_router(chat.router, tags=["Intelligence"])
api_router.include_router(documents.router, tags=["Documents"])
