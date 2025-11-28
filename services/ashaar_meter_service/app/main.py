from __future__ import annotations

from fastapi import FastAPI

from .api import router
from .config import get_settings
from .logging import configure_logging, get_logger


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)
    logger = get_logger("ashaar_meter_service")

    app = FastAPI(title="Shaer Ashaar Meter Service", version="0.1.0")

    @app.on_event("startup")
    async def _startup() -> None:
        from .api import get_ashaar_service

        service = get_ashaar_service(settings=settings)  # type: ignore[arg-type]
        if service.assets_loaded:
            logger.info("Ashaar meter service ready (assets available)")
        else:
            logger.warning("Ashaar meter service started without Ashaar assets")

    app.include_router(router)
    return app


app = create_app()
