from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from . import api
from .clients import get_client
from .config import get_settings
from .logging import configure_logging, get_logger


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)
    logger = get_logger("agent_service")

    app = FastAPI(title="Agent Service", version="0.1.0")

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        logger.info("Shutting down agent service")
        await get_client().close()

    app.include_router(api.router)
    return app


app = create_app()
