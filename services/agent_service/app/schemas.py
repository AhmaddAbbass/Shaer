from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str = Field(..., description="Message role (user/assistant/system)")
    content: str = Field(..., description="Message text")


class ChatRequest(BaseModel):
    messages: List[ChatMessage] = Field(default_factory=list)
    mode: Optional[str] = Field(default=None, description="UI mode: generate | fix | search | explain")


class PoemVersionOut(BaseModel):
    spec: Optional[Dict[str, Any]] = None
    verses: List[str] = Field(default_factory=list)


class ChatResponse(BaseModel):
    reply: str
    mode: Optional[str] = None
    poem_spec: Optional[Dict[str, Any]] = None
    poem_version: Optional[PoemVersionOut] = None
    agent_trace: Optional[List[Any]] = None
    library_context: Optional[List[Any]] = None


class ProxyRequest(BaseModel):
    payload: Dict[str, Any] = Field(default_factory=dict, description="Raw JSON payload to forward")
    target: Optional[str] = Field(
        default=None,
        description="Optional override path on orchestrator (e.g., /poem/generate). If omitted, uses /poem/generate.",
    )


class ProxyResponse(BaseModel):
    data: Dict[str, Any]
