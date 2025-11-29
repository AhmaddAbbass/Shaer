from __future__ import annotations

from typing import Any

import httpx

from services.shaer_service.app.schemas import BaytGenerationRequest


class ShaerServiceClient:
    def __init__(self, base_url: str, timeout: float) -> None:
        self._client = httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=timeout)

    async def close(self) -> None:
        await self._client.aclose()

    async def generate_bayt(self, request: BaytGenerationRequest) -> str:
        payload: dict[str, Any] = request.model_dump()
        response = await self._client.post("/generate-bayt", json=payload)
        response.raise_for_status()
        data = response.json()
        verse_text = data.get("verse_text")
        if not isinstance(verse_text, str):
            raise RuntimeError("Invalid response from shaer_service")
        return verse_text.strip()
