"""FastAPI application factory and lifespan."""

import logging
import threading
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import Settings, get_settings
from app.routers import health, predict
from app.services.classifier import get_classifier

logger = logging.getLogger(__name__)


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


def _make_lifespan(preload_model: bool):
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        settings = get_settings()
        _configure_logging(settings.log_level)

        app.state.classifier = None
        app.state.model_load_error = None

        if preload_model:
            classifier = get_classifier(settings)
            app.state.classifier = classifier
            logger.info(
                "Starting food-vision service (backend=%s, model=%s)",
                settings.inference_backend,
                settings.model_id,
            )

            def _load_model() -> None:
                try:
                    classifier.load()
                except Exception as exc:
                    app.state.model_load_error = str(exc)
                    logger.exception("Background model load failed")

            thread = threading.Thread(target=_load_model, name="model-loader", daemon=True)
            thread.start()
        else:
            logger.debug("Skipping model preload (test mode)")

        yield

        logger.info("Shutting down food-vision service")
        app.state.classifier = None
        app.state.model_load_error = None

    return lifespan


def create_app(settings: Settings | None = None, *, preload_model: bool = True) -> FastAPI:
    """Create and configure the FastAPI application."""
    cfg = settings or get_settings()

    docs_url = "/docs" if cfg.enable_docs else None
    redoc_url = "/redoc" if cfg.enable_docs else None
    openapi_url = "/openapi.json" if cfg.enable_docs else None

    app = FastAPI(
        title="OQ Food Vision",
        description="Food image classification for OQ Agile Athletes",
        version="1.0.0",
        lifespan=_make_lifespan(preload_model),
        docs_url=docs_url,
        redoc_url=redoc_url,
        openapi_url=openapi_url,
    )

    origins = [o.strip() for o in cfg.cors_origins.split(",") if o.strip()]
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["POST", "GET"],
            allow_headers=["*"],
        )

    app.include_router(health.router)
    app.include_router(predict.router)

    return app


app = create_app()
