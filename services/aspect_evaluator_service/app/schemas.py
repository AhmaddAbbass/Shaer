from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

from services.common_schemas.schemas import PoemSpec, YehiaFeedback

from .prompts import ASPECTS


class AspectJudgeResult(BaseModel):
    score_0_1: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Normalized numeric score; None if scoring disabled",
    )
    notes: str = Field(default="", description="Short explanation of the score")


class AspectEvaluation(BaseModel):
    yehia_feedback: YehiaFeedback
    judge: AspectJudgeResult


class EvaluateBaytRequest(BaseModel):
    verse_text: str = Field(..., min_length=1)
    spec: PoemSpec
    previous_verses: List[str] = Field(default_factory=list)
    use_openai_scoring: bool = Field(default=True)

    @field_validator("verse_text")
    @classmethod
    def _normalize_verse(cls, value: str) -> str:
        for line in value.splitlines():
            stripped = line.strip()
            if stripped:
                return stripped
        return value.strip()


class EvaluateBaytResponse(BaseModel):
    meaning: AspectEvaluation
    cohesion: AspectEvaluation
    fluency: AspectEvaluation
    poeticness: AspectEvaluation


class EvaluateAspectRequest(BaseModel):
    verse_text: str
    spec: PoemSpec
    previous_verses: List[str] = Field(default_factory=list)
    aspect: str = Field(..., description="One of the supported aspects")
    use_openai_scoring: bool = Field(default=True)

    @field_validator("aspect")
    @classmethod
    def _validate_aspect(cls, value: str) -> str:
        if value.lower() not in ASPECTS:
            raise ValueError(f"aspect must be one of {ASPECTS}")
        return value.lower()

    @field_validator("verse_text")
    @classmethod
    def _normalize(cls, value: str) -> str:
        for line in value.splitlines():
            stripped = line.strip()
            if stripped:
                return stripped
        return value.strip()


class EvaluateAspectResponse(BaseModel):
    aspect: str
    evaluation: AspectEvaluation
