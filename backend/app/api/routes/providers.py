
from fastapi import APIRouter, Depends

from backend.app.api.dependencies import get_provider_manager
from backend.app.intelligence.provider_manager import ProviderManager


router = APIRouter()


@router.get("/providers")
async def list_providers(
    manager: ProviderManager = Depends(get_provider_manager),
):
    """Return registered and initialized AI providers."""

    return {
        "default_provider": manager.get_default_provider_name(),
        "available_providers": manager.available_providers(),
        "initialized_providers": manager.initialized_providers(),
    }


@router.get("/providers/health")
async def provider_health(
    manager: ProviderManager = Depends(get_provider_manager),
):
    """Return live connectivity health for registered AI providers."""

    return await manager.check_all_provider_health()
