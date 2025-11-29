from __future__ import annotations

from typing import Any, Dict

import httpx

from services.aspect_evaluator_service.app.schemas import EvaluateBaytResponse
from services.common_schemas.schemas import BaytMeterEval, PoemSpec


class BaseServiceClient:
    def __init__(self, base_url: str, timeout: float) -> None:
        self._client = httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=timeout)

    async def close(self) -> None:
        await self._client.aclose()


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


class AspectEvaluatorClient(BaseServiceClient):
    async def evaluate_bayt(
        self,
        verse_text: str,
        spec: PoemSpec,
        previous_verses: list[str],
        use_openai_scoring: bool,
    ) -> EvaluateBaytResponse:
        payload: Dict[str, Any] = {
            "verse_text": verse_text,
            "spec": spec.model_dump(),
            "previous_verses": previous_verses,
            "use_openai_scoring": use_openai_scoring,
        }
        response = await self._client.post("/evaluate-bayt", json=payload)
        response.raise_for_status()
        data = response.json()
        return EvaluateBaytResponse.model_validate(data)
