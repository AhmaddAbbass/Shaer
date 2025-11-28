from __future__ import annotations

from fastapi import FastAPI

from . import api
from .client import get_client
from .config import get_settings
from .logging import configure_logging, get_logger


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)
    logger = get_logger("yehia_service")

    app = FastAPI(title="Yehia Service", version="0.1.0")
    app.include_router(api.router)

    @app.on_event("startup")
    async def _startup_event() -> None:
        # Validate env/config early
        get_client()
        logger.info("Yehia service ready")

    return app


app = create_app()
