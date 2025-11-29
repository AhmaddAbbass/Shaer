from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

from services.common_schemas.schemas import PoemSpec, RagSearchResponse
from services.scoring_service.app.schemas import ScoreResponse


class AgentStep(BaseModel):
    step: int
    agent: str
    tool: str
    summary: str


class GeneratePoemRequest(BaseModel):
    user_query: str = Field(..., min_length=1)
    desired_num_verses: Optional[int] = Field(default=None, ge=1, le=12)
    use_rag: bool = Field(default=True)
    use_openai_scoring: bool = Field(default=True)


class VerseWithScore(BaseModel):
    text: str
    scoring: ScoreResponse


class GeneratePoemResponse(BaseModel):
    spec: PoemSpec
    verses: List[VerseWithScore]
    rag_hits: Optional[List[str]] = None
    agent_trace: Optional[List[AgentStep]] = None


class FixBaytRequest(BaseModel):
    verse_text: str = Field(..., min_length=1)
    spec: Optional[PoemSpec] = None
    previous_verses: List[str] = Field(default_factory=list)
    use_openai_scoring: bool = Field(default=True)


class FixBaytResponse(BaseModel):
    verse_text: str
    scoring: ScoreResponse
    agent_trace: Optional[List[AgentStep]] = None


class ScoreBaytRequest(BaseModel):
    verse_text: str
    spec: Optional[PoemSpec] = None
    previous_verses: List[str] = Field(default_factory=list)
    use_openai_scoring: bool = Field(default=True)


class ScoreBaytResponse(BaseModel):
    scoring: ScoreResponse


class LibrarySearchRequest(BaseModel):
    query_text: str
    top_k: int = Field(default=5, ge=1, le=20)


class LibrarySearchResponse(BaseModel):
    results: RagSearchResponse
