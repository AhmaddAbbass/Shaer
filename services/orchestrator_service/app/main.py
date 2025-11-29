from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from .clients.enhancer_client import EnhancerServiceClient
from .clients.rag_client import RagServiceClient
from .clients.scoring_client import ScoringServiceClient
from .clients.shaer_client import ShaerServiceClient
from .clients.yehia_client import YehiaServiceClient
from .context import ServiceRegistry
from .logging import configure_logging, get_logger
from .routers import bayt_api, library_api, poem_api
from .settings import get_settings


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)
    logger = get_logger("orchestrator_service")

    app = FastAPI(title="Orchestrator Service", version="0.1.0")

    rag_client = RagServiceClient(settings.rag_service_url, settings.http_timeout_seconds)
    yehia_client = YehiaServiceClient(settings.yehia_service_url, settings.http_timeout_seconds)
    shaer_client = ShaerServiceClient(settings.shaer_service_url, settings.http_timeout_seconds)
    scoring_client = ScoringServiceClient(settings.scoring_service_url, settings.http_timeout_seconds)
    enhancer_client = EnhancerServiceClient(settings.enhancer_service_url, settings.http_timeout_seconds)

    app.state.registry = ServiceRegistry(
        settings=settings,
        rag_client=rag_client,
        yehia_client=yehia_client,
        shaer_client=shaer_client,
        scoring_client=scoring_client,
        enhancer_client=enhancer_client,
    )

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        logger.info("Shutting down orchestrator service")
        await rag_client.close()
        await yehia_client.close()
        await shaer_client.close()
        await scoring_client.close()
        await enhancer_client.close()

    app.include_router(poem_api.router)
    app.include_router(bayt_api.router)
    app.include_router(library_api.router)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
