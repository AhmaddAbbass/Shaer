from __future__ import annotations

from pydantic import BaseModel, Field


class EvalAshaarRequest(BaseModel):
    verse_text: str = Field(..., min_length=1, description="Full bayt text (optionally with [sep]).")


class AshaarScore(BaseModel):
    reward: float = Field(..., ge=0.0, le=1.0, description="Weighted Ashaar reward in [0, 1].")
    ashaar_score: float = Field(..., ge=0.0, le=1.0, description="Raw Ashaar structural score in [0, 1].")
    notes: str = Field(..., description="Diagnostic text about the Ashaar result.")


class EvalAshaarResponse(BaseModel):
    result: AshaarScore


class HealthResponse(BaseModel):
    status: str = Field(default="ok")
    assets_loaded: bool = Field(default=False)
