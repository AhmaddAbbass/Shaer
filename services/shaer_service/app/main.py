from fastapi import FastAPI

from . import api
from .client import get_client
from .logging import configure_logging


def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance."""
    configure_logging()
    app = FastAPI(title="Shaer Service", version="0.1.0")
    app.include_router(api.router)

    @app.on_event("startup")
    async def _startup_event() -> None:
        # Touch the client during startup to validate config early.
        get_client()

    @app.get("/health")
    async def healthcheck() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
