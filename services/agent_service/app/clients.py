from __future__ import annotations

from typing import List, Optional

import httpx

from services.common_schemas.schemas import (
    BaytMeterEval,
    LLMMessage,
    PoemSpec,
    RagPoemRecord,
    RagSearchHit,
    RagSearchResponse,
    YehiaFeedback,
)


class BaseServiceClient:
    def __init__(self, base_url: str, timeout: float):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=timeout)

    async def close(self) -> None:
        await self._client.aclose()


class YehiaServiceClient(BaseServiceClient):
    async def build_spec(self, user_query: str, rag_hits: List[RagSearchHit]) -> PoemSpec:
        payload = {
            "user_query": user_query,
            "rag_hits": [hit.model_dump() for hit in rag_hits],
        }
        response = await self._client.post("/build-spec", json=payload)
        response.raise_for_status()
        data = response.json()
        spec_data = data.get("spec")
        if not spec_data:
            raise RuntimeError("Invalid response from Yehia build-spec")
        return PoemSpec.model_validate(spec_data)

    async def feedback(self, verse_text: str, spec: PoemSpec) -> YehiaFeedback:
        payload = {
            "verse_text": verse_text,
            "spec": spec.model_dump(),
        }
        response = await self._client.post("/feedback", json=payload)
        response.raise_for_status()
        data = response.json()
        feedback = data.get("feedback")
        if not feedback:
            raise RuntimeError("Invalid response from Yehia feedback")
        return YehiaFeedback.model_validate(feedback)

    async def chat(self, messages: List[LLMMessage]) -> str:
        payload = {
            "messages": [msg.model_dump() for msg in messages],
        }
        response = await self._client.post("/chat", json=payload)
        response.raise_for_status()
        data = response.json()
        text = data.get("text")
        if not isinstance(text, str):
            raise RuntimeError("Invalid chat response from Yehia")
        return text.strip()


class ShaerServiceClient(BaseServiceClient):
    async def generate_bayt(self, spec: PoemSpec, sequence_number: int, previous_verses: List[str]) -> str:
        payload = {
            "poem_meter": spec.poem_meter,
            "poem_description": spec.poem_description,
            "num_verses": spec.num_verses,
            "sequence_number": sequence_number,
            "previous_verses": previous_verses,
            "poem_era": spec.poem_era,
            "poet_name": spec.poet_name,
            "poem_title": spec.poem_title,
        }
        response = await self._client.post("/generate-bayt", json=payload)
        response.raise_for_status()
        data = response.json()
        verse_text = data.get("verse_text")
        if not isinstance(verse_text, str):
            raise RuntimeError("Invalid response from Shaer generate-bayt")
        return verse_text.strip()


class RagServiceClient(BaseServiceClient):
    async def search(self, text: str, top_k: int) -> RagSearchResponse:
        params = {"text": text, "top_k": top_k}
        response = await self._client.get("/search", params=params)
        response.raise_for_status()
        data = response.json()
        return RagSearchResponse.model_validate(data)

    async def get_poem(self, poem_id: str) -> Optional[RagPoemRecord]:
        response = await self._client.get(f"/poem/{poem_id}")
        if response.status_code == 404:
            return None
        response.raise_for_status()
        data = response.json()
        return RagPoemRecord.model_validate(data)


class MeterServiceClient(BaseServiceClient):
    async def eval_bayt(self, verse_text: str, target_meter: str) -> BaytMeterEval:
        payload = {"verse_text": verse_text, "target_meter": target_meter}
        response = await self._client.post("/eval-bayt", json=payload)
        response.raise_for_status()
        data = response.json()
        result = data.get("result")
        if not result:
            raise RuntimeError("Invalid response from meter service")
        return BaytMeterEval.model_validate(result)
