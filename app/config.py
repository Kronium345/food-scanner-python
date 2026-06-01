"""Application configuration from environment variables."""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Service settings loaded from environment and optional .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    food_vision_api_key: str = Field(..., min_length=8)
    port: int = 8000
    log_level: str = "info"

    # onnx: lightweight CPU inference (recommended). torch: legacy PyTorch/transformers.
    inference_backend: Literal["onnx", "torch"] = "onnx"
    model_id: str = "onnx-community/swin-finetuned-food101-ONNX"
    onnx_model_file: str = "onnx/model_int8.onnx"
    top_k: int = Field(default=5, ge=1, le=10)
    # Drop low-confidence alternates from `concepts` (primaryConcept is always top-1).
    min_confidence: float = Field(default=0.15, ge=0.0, le=1.0)
    max_base64_chars: int = Field(default=900_000, ge=1)
    inference_timeout_sec: float = Field(default=55.0, gt=0)

    enable_docs: bool = True

    # Optional CORS — comma-separated origins; empty disables CORS middleware
    cors_origins: str = ""


@lru_cache
def get_settings() -> Settings:
    return Settings()
