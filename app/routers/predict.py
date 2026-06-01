"""Food prediction endpoint."""

import asyncio
import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.auth import verify_api_key
from app.config import Settings, get_settings
from app.models.schemas import PredictRequest, PredictResponse
from app.services.classifier import (
    FoodClassifier,
    decode_base64_image,
    get_classifier,
    strip_data_url_prefix,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["predict"])


@router.post(
    "/predict",
    response_model=PredictResponse,
    dependencies=[Depends(verify_api_key)],
)
async def predict(
    body: PredictRequest,
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
) -> PredictResponse:
    stripped = strip_data_url_prefix(body.image_base64)

    if not stripped:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="imageBase64 is required",
        )

    if len(stripped) > settings.max_base64_chars:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"Image payload exceeds maximum size of "
                f"{settings.max_base64_chars} base64 characters"
            ),
        )

    try:
        image = decode_base64_image(stripped)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    classifier: FoodClassifier = getattr(request.app.state, "classifier", None) or get_classifier(
        settings
    )

    if not classifier.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is still loading",
        )

    try:
        concepts, inference_ms = await asyncio.wait_for(
            asyncio.to_thread(classifier.predict, image, settings.top_k),
            timeout=settings.inference_timeout_sec,
        )
    except asyncio.TimeoutError as exc:
        logger.error("Inference timed out after %ss", settings.inference_timeout_sec)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Inference timed out",
        ) from exc
    except Exception as exc:
        logger.exception("Inference failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Inference failed",
        ) from exc

    return PredictResponse(
        concepts=concepts,
        model=classifier.model_id,
        inferenceMs=inference_ms,
    )
