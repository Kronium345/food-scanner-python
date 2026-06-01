"""API key authentication for /v1/* routes."""

from typing import Annotated

from fastapi import Depends, Header, HTTPException, status

from app.config import Settings, get_settings


def verify_api_key(
    settings: Annotated[Settings, Depends(get_settings)],
    x_food_vision_key: Annotated[str | None, Header()] = None,
    authorization: Annotated[str | None, Header()] = None,
) -> None:
    """Require X-Food-Vision-Key or Authorization: Bearer <key>."""
    provided: str | None = None

    if x_food_vision_key:
        provided = x_food_vision_key.strip()
    elif authorization:
        parts = authorization.split(None, 1)
        if len(parts) == 2 and parts[0].lower() == "bearer":
            provided = parts[1].strip()

    if not provided or provided != settings.food_vision_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized",
        )
