from __future__ import annotations

from typing import Any, Dict

import httpx

from services.common_schemas.schemas import PoemSpec, YehiaFeedback


class YehiaServiceClient:
    def __init__(self, base_url: str, timeout: float) -> None:
        self._client = httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=timeout)

    async def close(self) -> None:
        await self._client.aclose()

    async def feedback(self, verse_text: str, spec: PoemSpec, aspect: str | None, previous_verses: list[str] | None = None) -> YehiaFeedback:
        payload: Dict[str, Any] = {
            "verse_text": verse_text,
            "spec": spec.model_dump(),
            "aspect": aspect,
            "previous_verses": previous_verses,
        }
        response = await self._client.post("/feedback", json=payload)
        response.raise_for_status()
        data = response.json()
        if "feedback" not in data:
            raise RuntimeError("Invalid response from yehia_service")
        return YehiaFeedback.model_validate(data["feedback"])
