
from __future__ import annotations

from fastapi import FastAPI

from .api import router
from .chroma_client import ChromaClient
from .config import get_settings
from .logging import configure_logging, get_logger
from .neo4j_client import Neo4jClient


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)
    logger = get_logger("rag_service")

    app = FastAPI(
        title="Shaer RAG Service",
        version="0.1.0",
    )

    # Shared dependencies (can be overridden in tests)
    app.state.settings = settings
    app.state.chroma_client = ChromaClient(settings)
    app.state.neo4j_client = Neo4jClient(settings)

    @app.on_event("shutdown")
    def _shutdown() -> None:
        logger.info("Shutting down service and closing clients")
        try:
            app.state.neo4j_client.close()
        except Exception:
            pass

    app.include_router(router)
    return app


app = create_app()
