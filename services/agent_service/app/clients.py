from __future__ import annotations

from typing import Any, Dict

import httpx

from .config import Settings, get_settings


class OrchestratorClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._client = httpx.AsyncClient(
            base_url=settings.orchestrator_base_url.rstrip("/"),
            timeout=settings.http_timeout_seconds,
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def forward_json(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        response = await self._client.post(path, json=payload)
        response.raise_for_status()
        return response.json()


_CLIENT: OrchestratorClient | None = None


def get_client() -> OrchestratorClient:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = OrchestratorClient(get_settings())
    return _CLIENT
