
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.intelligence.provider_manager import ProviderManager


def test_providers_endpoint():
    with TestClient(app) as client:
        response = client.get("/api/v1/providers")

        assert response.status_code == 200

        data = response.json()

        assert data["available_providers"] == ["gemini", "groq"]
        assert data["default_provider"] in ["gemini", "groq"]
        assert data["initialized_providers"] == []


def test_provider_health_endpoint(monkeypatch):
    async def fake_health_check(self):
        return {
            "default_provider": "gemini",
            "providers": {
                "gemini": {
                    "provider": "gemini",
                    "status": "connected",
                    "configured": True,
                    "initialized": False,
                    "connected": True,
                    "message": "Provider connection verified.",
                },
                "groq": {
                    "provider": "groq",
                    "status": "not_configured",
                    "configured": False,
                    "initialized": False,
                    "connected": None,
                    "message": "Provider credentials or model are not configured.",
                },
            },
        }

    monkeypatch.setattr(
        ProviderManager,
        "check_all_provider_health",
        fake_health_check,
    )

    with TestClient(app) as client:
        response = client.get("/api/v1/providers/health")

        assert response.status_code == 200

        data = response.json()

        assert data["default_provider"] == "gemini"

        assert data["providers"]["gemini"]["status"] == "connected"
        assert data["providers"]["gemini"]["connected"] is True

        assert data["providers"]["groq"]["status"] == "not_configured"
        assert data["providers"]["groq"]["connected"] is None
