from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from . import api
from .clients import YehiaServiceClient
from .config import get_settings
from .evaluator import AspectEvaluator
from .logging import configure_logging, get_logger
from .openai_judge import AspectJudge


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)
    logger = get_logger("aspect_evaluator_service")

    app = FastAPI(title="Aspect Evaluator Service", version="0.1.0")
    yehia_client = YehiaServiceClient(settings.yehia_base_url, settings.request_timeout_seconds)
    judge = AspectJudge(settings)
    evaluator = AspectEvaluator(yehia_client, judge)
    app.state.aspect_evaluator = evaluator
    app.state.yehia_client = yehia_client

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        logger.info("Shutting down aspect evaluator service")
        await yehia_client.close()

    app.include_router(api.router)
    return app


app = create_app()
