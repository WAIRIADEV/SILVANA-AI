from fastapi.testclient import TestClient

from backend.app.main import app


def test_providers_endpoint():
    with TestClient(app) as client:
        response = client.get("/api/v1/providers")

        assert response.status_code == 200

        data = response.json()

        assert data["available_providers"] == ["gemini", "groq"]
        assert data["default_provider"] in ["gemini", "groq"]
        assert data["initialized_providers"] == []
