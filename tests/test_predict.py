"""Predict endpoint and image utility tests."""

import base64
from io import BytesIO
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.config import Settings
from app.services.classifier import (
    decode_base64_image,
    normalize_label,
    strip_data_url_prefix,
)


def test_strip_data_url_prefix() -> None:
    raw = base64.b64encode(b"hello").decode()
    prefixed = f"data:image/jpeg;base64,{raw}"
    assert strip_data_url_prefix(prefixed) == raw
    assert strip_data_url_prefix(raw) == raw


def test_normalize_label() -> None:
    assert normalize_label("apple_pie") == "apple pie"
    assert normalize_label("  Pizza  ") == "pizza"


def test_decode_base64_image_rgb() -> None:
    buf = BytesIO()
    Image.new("RGB", (8, 8), color=(255, 0, 0)).save(buf, format="JPEG")
    encoded = base64.b64encode(buf.getvalue()).decode()
    image = decode_base64_image(encoded)
    assert image.mode == "RGB"
    assert image.size == (8, 8)


def test_predict_requires_auth(client: TestClient, pizza_base64: str) -> None:
    response = client.post("/v1/predict", json={"imageBase64": pizza_base64})
    assert response.status_code == 401
    assert response.json() == {"detail": "Unauthorized"}


def test_predict_wrong_key(client: TestClient, pizza_base64: str) -> None:
    response = client.post(
        "/v1/predict",
        json={"imageBase64": pizza_base64},
        headers={"X-Food-Vision-Key": "wrong-key"},
    )
    assert response.status_code == 401


def test_predict_bearer_auth(
    client: TestClient, pizza_base64: str, settings: Settings
) -> None:
    response = client.post(
        "/v1/predict",
        json={"imageBase64": pizza_base64},
        headers={"Authorization": f"Bearer {settings.food_vision_api_key}"},
    )
    assert response.status_code == 200


def test_predict_success(
    client: TestClient, pizza_base64: str, auth_headers: dict[str, str], settings: Settings
) -> None:
    response = client.post(
        "/v1/predict",
        json={"imageBase64": pizza_base64},
        headers=auth_headers,
    )
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == settings.model_id
    assert body["inferenceMs"] == 42
    assert len(body["concepts"]) >= 1
    assert body["concepts"][0]["name"] == "pizza"
    assert body["concepts"][0]["confidence"] > 0.5


def test_predict_accepts_data_url_prefix(
    client: TestClient, pizza_base64: str, auth_headers: dict[str, str]
) -> None:
    response = client.post(
        "/v1/predict",
        json={"imageBase64": f"data:image/jpeg;base64,{pizza_base64}"},
        headers=auth_headers,
    )
    assert response.status_code == 200


def test_predict_rejects_invalid_base64(client: TestClient, auth_headers: dict[str, str]) -> None:
    response = client.post(
        "/v1/predict",
        json={"imageBase64": "not-valid-base64!!!"},
        headers=auth_headers,
    )
    assert response.status_code == 422
    assert "detail" in response.json()


def test_predict_rejects_oversized_payload(client: TestClient, auth_headers: dict[str, str]) -> None:
    huge = "A" * 900_001
    response = client.post(
        "/v1/predict",
        json={"imageBase64": huge},
        headers=auth_headers,
    )
    assert response.status_code == 413


def test_predict_model_loading(client: TestClient, pizza_base64: str, auth_headers: dict[str, str]) -> None:
    client.app.state.classifier._loaded = False
    response = client.post(
        "/v1/predict",
        json={"imageBase64": pizza_base64},
        headers=auth_headers,
    )
    assert response.status_code == 503
    assert response.json()["detail"] == "Model is still loading"


@pytest.mark.integration
def test_predict_integration_real_model(pizza_base64: str) -> None:
    """Optional: run against real ONNX model (requires network to download weights)."""
    pytest.importorskip("onnxruntime")
    from app.config import get_settings
    from app.main import create_app
    from app.services.classifier import get_classifier, reset_classifier

    get_settings.cache_clear()
    reset_classifier()

    settings = get_settings()
    app = create_app(settings, preload_model=False)
    classifier = get_classifier(settings)
    classifier.load()
    app.state.classifier = classifier

    with TestClient(app, raise_server_exceptions=True) as client:
        response = client.post(
            "/v1/predict",
            json={"imageBase64": pizza_base64},
            headers={"X-Food-Vision-Key": settings.food_vision_api_key},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["concepts"][0]["confidence"] > 0.1
