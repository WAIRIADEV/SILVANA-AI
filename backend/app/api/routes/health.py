from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health_check():
    return {
        "status": "operational",
        "application": "SILVANA",
        "version": "0.1.0",
        "message": "SILVANA systems are online."
    }
