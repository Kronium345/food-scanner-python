"""ONNX Runtime food classifier — lightweight alternative to PyTorch."""

from __future__ import annotations

import json
import logging
import threading
import time

import numpy as np
from PIL import Image

from app.config import Settings
from app.services.classifier import FoodClassifier, normalize_label

logger = logging.getLogger(__name__)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
RESCALE_FACTOR = 1.0 / 255.0
INPUT_SIZE = (224, 224)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


def _preprocess_image(image: Image.Image) -> np.ndarray:
    resized = image.convert("RGB").resize(INPUT_SIZE, Image.Resampling.BILINEAR)
    pixels = np.asarray(resized, dtype=np.float32) * RESCALE_FACTOR
    pixels = (pixels - IMAGENET_MEAN) / IMAGENET_STD
    pixels = pixels.transpose(2, 0, 1)
    return np.expand_dims(pixels, axis=0)


class OnnxFoodClassifier(FoodClassifier):
    """Food-101 classifier using ONNX Runtime on CPU."""

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self._session = None
        self._input_name: str | None = None
        self._id2label: dict[int, str] = {}
        self._lock = threading.Lock()

    def load(self) -> None:
        with self._lock:
            if self._loaded:
                return

            logger.info(
                "Loading ONNX model %s (%s)",
                self._settings.model_id,
                self._settings.onnx_model_file,
            )

            try:
                import onnxruntime as ort
                from huggingface_hub import hf_hub_download

                config_path = hf_hub_download(
                    repo_id=self._settings.model_id,
                    filename="config.json",
                )
                with open(config_path, encoding="utf-8") as handle:
                    config = json.load(handle)
                self._id2label = {int(key): value for key, value in config["id2label"].items()}

                model_path = hf_hub_download(
                    repo_id=self._settings.model_id,
                    filename=self._settings.onnx_model_file,
                )
                self._session = ort.InferenceSession(
                    model_path,
                    providers=["CPUExecutionProvider"],
                )
                self._input_name = self._session.get_inputs()[0].name

                dummy = Image.new("RGB", INPUT_SIZE, color=(128, 128, 128))
                self._run_inference(dummy)

                self._loaded = True
                self._load_error = None
                logger.info("ONNX model %s loaded and warmed up", self._settings.model_id)
            except Exception as exc:
                self._load_error = str(exc)
                logger.exception("Failed to load ONNX model")
                raise

    def _run_inference(self, image: Image.Image) -> np.ndarray:
        if self._session is None or self._input_name is None:
            raise RuntimeError("Model is not loaded")

        inputs = _preprocess_image(image)
        outputs = self._session.run(None, {self._input_name: inputs})
        logits = np.asarray(outputs[0], dtype=np.float32).squeeze()
        return _softmax(logits)

    def predict(
        self, image: Image.Image, top_k: int | None = None
    ) -> tuple[list[dict[str, float | str]], int]:
        if not self._loaded:
            raise RuntimeError("Model is not loaded")

        k = top_k if top_k is not None else self._settings.top_k
        k = max(1, min(k, 10))

        start = time.perf_counter()
        probabilities = self._run_inference(image)
        elapsed_ms = int((time.perf_counter() - start) * 1000)

        top_indices = np.argsort(probabilities)[::-1][:k]
        concepts: list[dict[str, float | str]] = []
        for index in top_indices:
            label = self._id2label.get(int(index), str(int(index)))
            concepts.append(
                {
                    "name": normalize_label(label),
                    "confidence": float(probabilities[index]),
                }
            )

        return concepts, elapsed_ms
