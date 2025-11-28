from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

from services.common_schemas.schemas import PoemSpec


class ChatMessage(BaseModel):
    id: str = Field(..., description="Frontend message identifier")
    role: Literal["user", "assistant"] = Field(...)
    content: str = Field(..., min_length=1)


class ChatRequest(BaseModel):
    messages: List[ChatMessage] = Field(..., min_length=1)
    mode: Optional[str] = Field(default=None)


class AgentStep(BaseModel):
    step: int
    agent: str
    tool: str
    summary: str


class LibraryItem(BaseModel):
    poem_id: str
    title: str
    poet_name: Optional[str] = None
    poem_meter: Optional[str] = None
    poem_era: Optional[str] = None
    poem_theme: Optional[str] = None
    snippet: Optional[str] = Field(default=None, description="Poem preview text")


class PoemVersion(BaseModel):
    spec: PoemSpec
    verses: List[str] = Field(default_factory=list)
    original_verses: Optional[List[str]] = None
    notes: Optional[str] = None


class ChatResponse(BaseModel):
    reply: str
    mode: Optional[str] = None
    poem_spec: Optional[PoemSpec] = None
    poem_version: Optional[PoemVersion] = None
    agent_trace: List[AgentStep] = Field(default_factory=list)
    library_context: List[LibraryItem] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    error_code: Optional[str] = None
