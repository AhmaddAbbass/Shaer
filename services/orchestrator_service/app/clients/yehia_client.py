from __future__ import annotations

from typing import Any, Dict, List

from services.common_schemas.schemas import LLMMessage, PoemSpec, RagSearchHit, YehiaFeedback

from .base import BaseClient


class YehiaServiceClient(BaseClient):
    async def build_spec(self, user_query: str, rag_hits: List[RagSearchHit]) -> PoemSpec:
        payload = {
            "user_query": user_query,
            "rag_hits": [hit.model_dump() for hit in rag_hits],
        }
        response = await self._client.post("/build-spec", json=payload)
        response.raise_for_status()
        data = response.json()
        return PoemSpec.model_validate(data["spec"])

    async def feedback(self, verse_text: str, spec: PoemSpec, aspect: str | None = None, previous_verses: list[str] | None = None) -> YehiaFeedback:
        payload: Dict[str, Any] = {
            "verse_text": verse_text,
            "spec": spec.model_dump(),
            "aspect": aspect,
            "previous_verses": previous_verses or [],
        }
        response = await self._client.post("/feedback", json=payload)
        response.raise_for_status()
        data = response.json()
        return YehiaFeedback.model_validate(data["feedback"])

    async def chat(self, messages: List[LLMMessage]) -> str:
        payload = {"messages": [msg.model_dump() for msg in messages]}
        response = await self._client.post("/chat", json=payload)
        response.raise_for_status()
        data = response.json()
        return str(data.get("text", "")).strip()
