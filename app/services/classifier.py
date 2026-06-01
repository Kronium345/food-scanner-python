"""Shared image utilities and classifier factory."""

from __future__ import annotations

import base64
import logging
import re
from abc import ABC, abstractmethod
from io import BytesIO

from PIL import Image, UnidentifiedImageError

from app.config import Settings

logger = logging.getLogger(__name__)

DATA_URL_PREFIX = re.compile(r"^data:image/[^;]+;base64,", re.IGNORECASE)


def strip_data_url_prefix(image_base64: str) -> str:
    return DATA_URL_PREFIX.sub("", image_base64.strip())


def normalize_label(label: str) -> str:
    """Normalize Food-101 style labels for Node/Clarifai compatibility."""
    return label.strip().lower().replace("_", " ")


def decode_base64_image(image_base64: str) -> Image.Image:
    """Decode base64 string to RGB PIL Image."""
    try:
        raw = base64.b64decode(image_base64, validate=True)
    except Exception as exc:
        raise ValueError("Invalid base64 encoding") from exc

    try:
        image = Image.open(BytesIO(raw))
        return image.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("Cannot decode image data") from exc


class FoodClassifier(ABC):
    """Abstract food classifier loaded once at startup."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._loaded = False
        self._load_error: str | None = None

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def load_error(self) -> str | None:
        return self._load_error

    @property
    def model_id(self) -> str:
        return self._settings.model_id

    @abstractmethod
    def load(self) -> None:
        """Load model weights into memory."""

    @abstractmethod
    def predict(
        self, image: Image.Image, top_k: int | None = None
    ) -> tuple[list[dict[str, float | str]], int]:
        """Run classification and return (concepts, inference_ms)."""


_classifier: FoodClassifier | None = None


def create_classifier(settings: Settings) -> FoodClassifier:
    if settings.inference_backend == "torch":
        from app.services.torch_classifier import TorchFoodClassifier

        return TorchFoodClassifier(settings)

    from app.services.onnx_classifier import OnnxFoodClassifier

    return OnnxFoodClassifier(settings)


def get_classifier(settings: Settings) -> FoodClassifier:
    global _classifier
    if _classifier is None:
        _classifier = create_classifier(settings)
    return _classifier


def reset_classifier() -> None:
    """Reset singleton — for tests only."""
    global _classifier
    _classifier = None
