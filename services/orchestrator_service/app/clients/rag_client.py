from __future__ import annotations

from typing import Any, Dict, Optional

from services.common_schemas.schemas import RagSearchResponse

from .base import BaseClient


class RagServiceClient(BaseClient):
    async def search(
        self,
        text: str,
        top_k: int,
        filters: Optional[Dict[str, str]] = None,
    ) -> RagSearchResponse:
        params: Dict[str, Any] = {"text": text, "top_k": top_k}
        if filters:
            params.update(filters)
        response = await self._client.get("/search", params=params)
        response.raise_for_status()
        data = response.json()
        return RagSearchResponse.model_validate(data)
