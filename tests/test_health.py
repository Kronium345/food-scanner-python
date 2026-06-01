"""Health endpoint tests."""

from fastapi.testclient import TestClient

from app.config import Settings


def test_health_returns_ok(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready_returns_ready_when_model_loaded(client: TestClient, settings: Settings) -> None:
    response = client.get("/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["model"] == settings.model_id
