from __future__ import annotations

from fastapi import FastAPI

from .api import router
from .config import get_settings
from .logging import configure_logging, get_logger


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)
    logger = get_logger("meter_service")

    app = FastAPI(title="Shaer Meter Service", version="0.1.0")

    # Warm up scansion assets on startup for health checks.
    @app.on_event("startup")
    async def _startup() -> None:
        from .api import get_scansion_service

        scansion = get_scansion_service(settings=settings)  # type: ignore[arg-type]
        if scansion.assets_loaded:
            logger.info("Meter service ready (assets loaded)")
        else:
            logger.warning("Meter service started but assets not loaded")

    app.include_router(router)
    return app


app = create_app()
