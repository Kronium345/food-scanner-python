"""Pydantic request and response models."""

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    image_base64: str = Field(..., min_length=1, alias="imageBase64")

    model_config = {"populate_by_name": True}


class Concept(BaseModel):
    name: str
    confidence: float = Field(..., ge=0.0, le=1.0)


class PredictResponse(BaseModel):
    """Vision result for Node: use primaryConcept for meal totals; concepts are filtered alternates."""

    primary_concept: Concept | None = Field(None, alias="primaryConcept")
    concepts: list[Concept]
    model: str
    inference_ms: int = Field(..., alias="inferenceMs")

    model_config = {"populate_by_name": True}


class HealthResponse(BaseModel):
    status: str = "ok"


class ReadyResponse(BaseModel):
    status: str
    model: str | None = None


class ErrorResponse(BaseModel):
    detail: str
