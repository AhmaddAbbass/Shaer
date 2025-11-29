from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from . import api
from .clients import ShaerServiceClient
from .config import get_settings
from .logging import configure_logging, get_logger
from .policy import EnhancementPlanner


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)
    logger = get_logger("enhancer_service")

    app = FastAPI(title="Enhancer Service", version="0.1.0")

    shaer_client = ShaerServiceClient(settings.shaer_base_url, settings.request_timeout_seconds)
    planner = EnhancementPlanner(settings)

    app.state.shaer_client = shaer_client
    app.state.enhancement_planner = planner

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        logger.info("Shutting down enhancer service")
        await shaer_client.close()

    app.include_router(api.router)
    return app


app = create_app()
