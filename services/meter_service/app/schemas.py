from __future__ import annotations

from pydantic import BaseModel, Field

from services.common_schemas.schemas import BaytMeterEval


class EvalBaytRequest(BaseModel):
    verse_text: str = Field(..., min_length=1, description="Full bayt as text")
    target_meter: str = Field(..., min_length=1, description="Arabic meter name expected for the bayt")


class EvalBaytResponse(BaseModel):
    result: BaytMeterEval


class HealthResponse(BaseModel):
    status: str = Field(default="ok")
    assets_loaded: bool = Field(default=False)
