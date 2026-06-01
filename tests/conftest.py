"""Pytest fixtures and test app factory."""

from __future__ import annotations

import base64
import os
from collections.abc import Generator
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.main import create_app
from app.services.classifier import FoodClassifier, reset_classifier

FIXTURES_DIR = Path(__file__).parent / "fixtures"
PIZZA_FIXTURE = FIXTURES_DIR / "pizza.jpg"


@pytest.fixture(scope="session", autouse=True)
def _test_env() -> None:
    os.environ.setdefault("ENABLE_DOCS", "false")
    get_settings.cache_clear()
    reset_classifier()


@pytest.fixture
def settings() -> Settings:
    get_settings.cache_clear()
    reset_classifier()
    return get_settings()


class MockClassifier(FoodClassifier):
    """Classifier stub that skips model download."""

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self._loaded = True

    def load(self) -> None:
        self._loaded = True

    def predict(self, image, top_k: int | None = None):  # noqa: ANN001
        k = top_k or self._settings.top_k
        concepts = [
            {"name": "pizza", "confidence": 0.94},
            {"name": "cheeseburger", "confidence": 0.03},
        ][:k]
        return concepts, 42


@pytest.fixture
def mock_classifier(settings: Settings) -> MockClassifier:
    reset_classifier()
    return MockClassifier(settings)


@pytest.fixture
def client(settings: Settings, mock_classifier: MockClassifier) -> Generator[TestClient, None, None]:
    get_settings.cache_clear()

    app = create_app(settings, preload_model=False)
    app.dependency_overrides[get_settings] = lambda: settings

    with TestClient(app, raise_server_exceptions=True) as test_client:
        test_client.app.state.classifier = mock_classifier
        yield test_client

    app.dependency_overrides.clear()


@pytest.fixture
def pizza_base64() -> str:
    if not PIZZA_FIXTURE.exists():
        pytest.skip("tests/fixtures/pizza.jpg not found — run scripts/download_fixture.py")
    return base64.b64encode(PIZZA_FIXTURE.read_bytes()).decode("ascii")


@pytest.fixture
def auth_headers(settings: Settings) -> dict[str, str]:
    return {"X-Food-Vision-Key": settings.food_vision_api_key}
