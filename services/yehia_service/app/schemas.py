from __future__ import annotations

from typing import List, Optional, Literal

from pydantic import BaseModel, Field, conint

from services.common_schemas.schemas import (
    LLMMessage,
    PoemSpec,
    RagSearchHit,
    YehiaFeedback,
)


class ChatRequest(BaseModel):
    messages: List[LLMMessage] = Field(..., description="Chat messages to send to Yehia")


class ChatResponse(BaseModel):
    text: str = Field(..., description="Generated text from Yehia")


class BuildSpecRequest(BaseModel):
    user_query: str = Field(..., min_length=1, description="User query in Arabic")
    rag_hits: List[RagSearchHit] = Field(default_factory=list, description="Optional RAG hits to guide spec")


class BuildSpecResponse(BaseModel):
    spec: PoemSpec


class FeedbackRequest(BaseModel):
    verse_text: str = Field(..., min_length=1, description="Bayt to critique")
    spec: PoemSpec
    aspect: Optional[Literal["meaning", "cohesion", "fluency", "poeticness"]] = Field(
        default=None,
        description="Optional evaluation aspect to guide Yehia's feedback",
    )
    previous_verses: List[str] = Field(
        default_factory=list,
        description="Optional previous verses to provide cohesion context",
    )


class FeedbackResponse(BaseModel):
    feedback: YehiaFeedback
