from fastapi import FastAPI
from backend.app.api.router import api_router

app = FastAPI(
    title="SILVANA",
    description="AI-powered intelligent assistant.",
    version="0.1.0"
)


@app.get("/")
async def root():
    return {
        "application": "SILVANA",
        "message": "Observe. Understand. Execute. Verify.",
        "documentation": "/docs"
    }


app.include_router(api_router, prefix="/api/v1")
