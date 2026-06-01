"""Health and readiness endpoints."""

from typing import Annotated

from fastapi import APIRouter, Depends, Request, status
from fastapi.responses import JSONResponse

from app.config import Settings, get_settings
from app.models.schemas import HealthResponse, ReadyResponse
from app.services.classifier import FoodClassifier, get_classifier

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.get("/ready", response_model=ReadyResponse)
async def ready(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
) -> JSONResponse | ReadyResponse:
    classifier: FoodClassifier = getattr(request.app.state, "classifier", None) or get_classifier(
        settings
    )

    load_error = getattr(request.app.state, "model_load_error", None) or classifier.load_error
    if load_error:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "error", "model": settings.model_id, "detail": load_error},
        )

    if classifier.is_loaded:
        return ReadyResponse(status="ready", model=classifier.model_id)

    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content=ReadyResponse(status="loading", model=settings.model_id).model_dump(),
    )
