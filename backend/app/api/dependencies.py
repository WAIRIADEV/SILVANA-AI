from fastapi import Request

from backend.app.intelligence.provider_manager import ProviderManager


def get_provider_manager(request: Request) -> ProviderManager:
    return request.app.state.provider_manager