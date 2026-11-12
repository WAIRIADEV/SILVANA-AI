from fastapi import APIRouter, Depends

from backend.app.api.dependencies import get_provider_manager
from backend.app.intelligence.provider_manager import ProviderManager

router = APIRouter()


@router.get("/providers")
async def list_providers(
    manager: ProviderManager = Depends(get_provider_manager),
):
    return {
        "default_provider": manager.get_default_provider_name(),
        "available_providers": manager.available_providers(),
        "initialized_providers": list(manager._providers.keys()),
    }
