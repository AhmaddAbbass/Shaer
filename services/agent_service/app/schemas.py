from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class ProxyRequest(BaseModel):
    payload: Dict[str, Any] = Field(default_factory=dict, description="Raw JSON payload to forward")
    target: Optional[str] = Field(
        default=None,
        description="Optional override path on orchestrator (e.g., /poem/generate). If omitted, uses /poem/generate.",
    )


class ProxyResponse(BaseModel):
    data: Dict[str, Any]
