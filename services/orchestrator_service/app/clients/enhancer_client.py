from __future__ import annotations

from typing import Any, Dict, List

from services.common_schemas.schemas import PoemSpec
from services.scoring_service.app.schemas import ScoreResponse

from .base import BaseClient


class EnhancerServiceClient(BaseClient):
    async def enhance_bayt(
        self,
        spec: PoemSpec,
        previous_verses: List[str],
        sequence_number: int,
        failed_verse: str,
        scoring: ScoreResponse,
        feedback_summary: List[str],
    ) -> str:
        # Pick the lowest-scoring aspect to pass richer feedback to the enhancer.
        aspect_map = {
            "meaning": scoring.aspects.meaning,
            "cohesion": scoring.aspects.cohesion,
            "fluency": scoring.aspects.fluency,
            "poeticness": scoring.aspects.poeticness,
        }
        worst_aspect_key = min(
            aspect_map.keys(),
            key=lambda k: (aspect_map[k].judge.score_0_1 or 0.0),
        )
        worst_feedback = aspect_map[worst_aspect_key].yehia_feedback

        payload: Dict[str, Any] = {
            "request": {
                "poem_meter": spec.poem_meter,
                "poem_description": spec.poem_description,
                "num_verses": spec.num_verses,
                "sequence_number": sequence_number,
                "previous_verses": previous_verses,
                "poem_era": spec.poem_era,
                "poet_name": spec.poet_name,
                "poem_title": spec.poem_title,
            },
            "failed_verse": failed_verse,
            "meter_eval": scoring.meter_eval.model_dump(),
            "yehia_feedback": worst_feedback.model_dump(),
            "feedback_summary": feedback_summary,
        }
        response = await self._client.post("/enhance-bayt", json=payload)
        response.raise_for_status()
        data = response.json()
        return str(data.get("candidate_verse", "")).strip()
