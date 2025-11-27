
# services/common_schemas/schemas.py

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------
# Core poem-generation objects
# ---------------------------------------------------------------

class PoemSpec(BaseModel):
    """
    Structured description of the poem we want to generate.
    This is what Yehia infers and what Shaer consumes.
    """

    poem_title: Optional[str] = Field(
        default=None,
        description="Optional title for the poem. Can be empty."
    )
    poem_meter: str = Field(
        ...,
        description="Target meter name in Arabic, e.g. 'البسيط', 'الكامل'."
    )
    poem_theme: Optional[str] = Field(
        default=None,
        description="Theme / purpose, e.g. 'غزل', 'رثاء', 'حماسة'."
    )
    poem_era: Optional[str] = Field(
        default=None,
        description="Historical era, e.g. 'العصر العباسي', 'العصر الحديث'."
    )
    poet_name: Optional[str] = Field(
        default=None,
        description="Name or style hint of poet to emulate (or None for generic style)."
    )
    poem_description: str = Field(
        ...,
        description="Short Arabic prose description of the overall poem content and mood."
    )
    num_verses: int = Field(
        ...,
        gt=0,
        description="Total number of bayts (verses) in the poem."
    )

    # optional extras – use if/when needed
    poem_language_type: Optional[str] = Field(
        default=None,
        description="e.g. 'فصحى', 'عامية', or other variety label."
    )
    style_notes: Optional[str] = Field(
        default=None,
        description="Extra style constraints, e.g. 'صور مكثفة', 'لغة بسيطة'."
    )


class Verse(BaseModel):
    """
    One bayt: صدر + عجز on a single line.
    """

    index: int = Field(
        ...,
        ge=1,
        description="1-based index of the bayt within the poem."
    )
    text: str = Field(
        ...,
        description="Full bayt as a single line: 'الصدر ... العجز'."
    )


class Poem(BaseModel):
    """
    A poem instance: spec + generated (or retrieved) verses.
    """

    spec: PoemSpec
    verses: List[Verse] = Field(
        default_factory=list,
        description="Ordered list of bayts."
    )

    # optional metadata (useful when the poem comes from RAG or a DB)
    poem_id: Optional[str] = Field(
        default=None,
        description="Stable identifier (e.g. from RAG / Neo4j)."
    )
    source_url: Optional[str] = Field(
        default=None,
        description="Original URL if this poem exists in an external corpus."
    )
    detected_meter: Optional[str] = Field(
        default=None,
        description="Meter detected by a classifier / scansion model, if different from spec."
    )
    rhyme: Optional[str] = Field(
        default=None,
        description="Optional rhyme / قافية marker."
    )


# ---------------------------------------------------------------
# Evaluation / feedback objects
# ---------------------------------------------------------------

class BaytMeterEval(BaseModel):
    """
    Output of the meter_service for a single bayt.
    """

    target_meter: str = Field(
        ...,
        description="The meter we want this bayt to follow."
    )
    meter_score: int = Field(
        ...,
        ge=0,
        le=100,
        description="0–100 score of how well the bayt matches the target meter."
    )
    on_meter: bool = Field(
        ...,
        description="True if the bayt is considered acceptable for the target meter."
    )
    notes: str = Field(
        ...,
        description="Short Arabic explanation (where the meter breaks, etc.)."
    )


class YehiaFeedback(BaseModel):
    """
    High-level feedback from Yehia about a single bayt,
    given the intended PoemSpec.
    """

    ok: bool = Field(
        ...,
        description="True if the bayt is acceptable w.r.t. meaning/style/instructions."
    )
    score: Optional[int] = Field(
        default=None,
        ge=0,
        le=100,
        description="Optional 0–100 quality score (alignment with spec, coherence, etc.)."
    )
    feedback: str = Field(
        ...,
        description="Arabic explanation and suggestions to improve the bayt."
    )


# ---------------------------------------------------------------
# RAG-related objects
# ---------------------------------------------------------------

class RagSearchRequest(BaseModel):
    """
    Input for semantic search in the RAG service.
    """

    query_text: str = Field(
        ...,
        description="User or agent query in Arabic, describing desired poem/topic."
    )
    top_k: int = Field(
        default=5,
        gt=0,
        description="Number of candidates to retrieve from Chroma."
    )


class RagSearchHit(BaseModel):
    """
    One search hit from Chroma + Neo4j.
    """

    poem_id: str = Field(
        ...,
        description="Identifier of the poem (matches Neo4j / HF dataset id)."
    )
    poem_title: Optional[str] = None
    poet_name: Optional[str] = None

    poem_description: str = Field(
        ...,
        description="Cleaned Arabic prose description used for retrieval."
    )

    poem_meter: Optional[str] = None
    poem_era: Optional[str] = None
    poem_theme: Optional[str] = None

    has_bad_description: bool = Field(
        default=False,
        description="True if description was flagged as low quality."
    )


class RagSearchResponse(BaseModel):
    """
    Wrapper for RAG search results.
    """

    hits: List[RagSearchHit] = Field(
        default_factory=list,
        description="Top-k poem candidates for the given query."
    )


class RagPoemRecord(BaseModel):
    """
    Full poem record fetched from the graph / DB.
    """

    poem_id: str
    poem_title: Optional[str] = None
    poet_name: Optional[str] = None
    poem_meter: Optional[str] = None
    poem_era: Optional[str] = None
    poem_theme: Optional[str] = None
    num_verses: int = Field(
        ...,
        ge=0,
        description="Total number of verses recorded in the corpus."
    )
    verses: List[str] = Field(
        default_factory=list,
        description="List of bayt strings in corpus order."
    )
    source_url: Optional[str] = None


# ---------------------------------------------------------------
# Generic LLM message shape
# ---------------------------------------------------------------

class LLMMessage(BaseModel):
    """
    Minimal chat message structure shared between services
    when building prompts for Yehia / Shaer.
    """

    role: str = Field(
        ...,
        description="One of: 'system', 'user', 'assistant'."
    )
    content: str = Field(
        ...,
        description="Text content of the message."
    )
