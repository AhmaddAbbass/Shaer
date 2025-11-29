from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from services.common_schemas.schemas import BaytMeterEval, YehiaFeedback
from services.shaer_service.app.schemas import BaytGenerationRequest


class EnhanceBaytRequest(BaseModel):
    request: BaytGenerationRequest = Field(..., description="Original Shaer request payload")
    failed_verse: str = Field(..., description="The previously generated verse that failed evaluation")
    meter_eval: Optional[BaytMeterEval] = Field(
        default=None,
        description="Meter evaluation result for the failed verse",
    )
    yehia_feedback: Optional[YehiaFeedback] = Field(
        default=None,
        description="Yehia feedback object for the failed verse",
    )
    feedback_summary: List[str] = Field(
        default_factory=list,
        description="1-3 short summary bullets provided by the orchestrator",
    )

    @field_validator("failed_verse", mode="before")
    @classmethod
    def _normalize_failed(cls, value: str) -> str:
        if not isinstance(value, str):
            return ""
        for line in value.splitlines():
            stripped = line.strip()
            if stripped:
                return stripped
        return ""

    @field_validator("feedback_summary", mode="before")
    @classmethod
    def _ensure_summary_list(cls, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            lines = [line.strip("-• \t") for line in value.splitlines() if line.strip()]
            return lines
        if isinstance(value, list):
            cleaned: List[str] = []
            for item in value:
                if isinstance(item, str) and item.strip():
                    cleaned.append(item.strip())
            return cleaned
        return []


class EnhanceBaytResponse(BaseModel):
    candidate_verse: str = Field(..., description="Improved single-bayt candidate")
    applied_changes: List[str] = Field(default_factory=list, description="What modifications were applied to the request")
    debug: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional debug metadata (not exposed to end users)",
    )
