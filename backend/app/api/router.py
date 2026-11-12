
from fastapi import APIRouter

from backend.app.api.routes import health, providers, chat

api_router = APIRouter()

api_router.include_router(health.router, tags=["System"])
api_router.include_router(providers.router, tags=["Intelligence"])
api_router.include_router(chat.router, tags=["Intelligence"])
