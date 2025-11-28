from __future__ import annotations

from fastapi import FastAPI

from .api import router
from .clients import MeterServiceClient, RagServiceClient, ShaerServiceClient, YehiaServiceClient
from .config import get_settings
from .logging import configure_logging, get_logger
from .orchestrator import OrchestratorService


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)
    logger = get_logger("agent_service")

    app = FastAPI(title="Shaer Agent Service", version="0.1.0")

    yehia_client = YehiaServiceClient(settings.yehia_base_url, settings.http_timeout_seconds)
    shaer_client = ShaerServiceClient(settings.shaer_base_url, settings.http_timeout_seconds)
    rag_client = RagServiceClient(settings.rag_base_url, settings.http_timeout_seconds)
    meter_client = MeterServiceClient(settings.meter_base_url, settings.http_timeout_seconds)

    orchestrator = OrchestratorService(settings, yehia_client, shaer_client, rag_client, meter_client)
    app.state.orchestrator = orchestrator

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        logger.info("Shutting down Shaer agent service")
        await orchestrator.aclose()

    app.include_router(router)
    return app


app = create_app()
