from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from . import api
from .aggregator import ScoringAggregator
from .clients import AspectEvaluatorClient, MeterServiceClient
from .config import get_settings
from .logging import configure_logging, get_logger
from .scoring import ScoreCalculator


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)
    logger = get_logger("scoring_service")

    app = FastAPI(title="Scoring Service", version="0.1.0")
    meter_client = MeterServiceClient(settings.meter_base_url, settings.http_timeout_seconds)
    aspect_client = AspectEvaluatorClient(settings.aspect_eval_base_url, settings.http_timeout_seconds)
    calculator = ScoreCalculator(settings)
    aggregator = ScoringAggregator(meter_client, aspect_client, calculator, settings)

    app.state.scoring_aggregator = aggregator
    app.state.meter_client = meter_client
    app.state.aspect_client = aspect_client

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        logger.info("Shutting down scoring service")
        await meter_client.close()
        await aspect_client.close()

    app.include_router(api.router)
    return app


app = create_app()
