from __future__ import annotations

from services.scoring_service.app.schemas import ScoreResponse
from services.common_schemas.schemas import PoemSpec

from .base import BaseClient


class ScoringServiceClient(BaseClient):
    async def score_bayt(
        self,
        verse_text: str,
        spec: PoemSpec,
        previous_verses: list[str],
        use_openai_scoring: bool = True,
    ) -> ScoreResponse:
        payload = {
            "verse_text": verse_text,
            "spec": spec.model_dump(),
            "previous_verses": previous_verses,
            "use_openai_scoring": use_openai_scoring,
        }
        response = await self._client.post("/score", json=payload)
        response.raise_for_status()
        data = response.json()
        return ScoreResponse.model_validate(data)
