from __future__ import annotations

from typing import Dict, List

from pydantic import BaseModel, Field

from services.aspect_evaluator_service.app.schemas import EvaluateBaytResponse
from services.common_schemas.schemas import BaytMeterEval, PoemSpec


class ScoreBaytRequest(BaseModel):
    verse_text: str = Field(..., min_length=1)
    spec: PoemSpec
    previous_verses: List[str] = Field(default_factory=list)
    use_openai_scoring: bool = Field(default=True)


class ScoreResponse(BaseModel):
    meter_eval: BaytMeterEval
    aspects: EvaluateBaytResponse
    final_score: float = Field(..., ge=0.0, le=1.0)
    passed: bool
    breakdown: Dict[str, float] = Field(default_factory=dict)
