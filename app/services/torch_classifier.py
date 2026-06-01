"""Legacy PyTorch/transformers classifier (requires Python 3.11–3.13 and torch)."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from PIL import Image

from app.config import Settings
from app.services.classifier import FoodClassifier, normalize_label

logger = logging.getLogger(__name__)


class TorchFoodClassifier(FoodClassifier):
    """Food classifier using Hugging Face transformers pipeline on CPU."""

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self._pipeline: Any | None = None
        self._lock = threading.Lock()

    def load(self) -> None:
        with self._lock:
            if self._loaded:
                return

            logger.info("Loading PyTorch model %s on CPU", self._settings.model_id)

            try:
                import torch
                from transformers import pipeline

                torch.set_num_threads(1)

                self._pipeline = pipeline(
                    "image-classification",
                    model=self._settings.model_id,
                    device=-1,
                )

                dummy = Image.new("RGB", (224, 224), color=(128, 128, 128))
                self._pipeline(dummy, top_k=1)

                self._loaded = True
                self._load_error = None
                logger.info("PyTorch model %s loaded and warmed up", self._settings.model_id)
            except Exception as exc:
                self._load_error = str(exc)
                logger.exception("Failed to load PyTorch model")
                raise

    def predict(
        self, image: Image.Image, top_k: int | None = None
    ) -> tuple[list[dict[str, float | str]], int]:
        if not self._loaded or self._pipeline is None:
            raise RuntimeError("Model is not loaded")

        k = top_k if top_k is not None else self._settings.top_k
        k = max(1, min(k, 10))

        start = time.perf_counter()
        raw_results = self._pipeline(image, top_k=k)
        elapsed_ms = int((time.perf_counter() - start) * 1000)

        concepts: list[dict[str, float | str]] = []
        for item in raw_results:
            concepts.append(
                {
                    "name": normalize_label(str(item["label"])),
                    "confidence": float(item["score"]),
                }
            )

        concepts.sort(key=lambda c: float(c["confidence"]), reverse=True)
        return concepts, elapsed_ms
