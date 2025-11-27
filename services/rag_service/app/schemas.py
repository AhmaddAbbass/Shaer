
from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field

from services.common_schemas.schemas import (
    RagPoemRecord,
    RagSearchHit,
    RagSearchResponse,
)


class HealthResponse(BaseModel):
    status: str = Field(default="ok")
    neo4j: str = Field(default="unknown")
    chroma: str = Field(default="unknown")


class FilterResponse(BaseModel):
    hits: List[RagSearchHit] = Field(default_factory=list)


class SimilarResponse(RagSearchResponse):
    pass
