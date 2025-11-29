from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from services.common_schemas.schemas import PoemSpec

from ..context import ServiceRegistry
from ..schemas import GeneratePoemRequest, GeneratePoemResponse, VerseWithScore
from ..settings import Settings
from ..utils.feedback import summarize_feedback

router = APIRouter(prefix="/poem", tags=["poem"])


def get_registry(request: Request) -> ServiceRegistry:
    registry = getattr(request.app.state, "registry", None)
    if registry is None:
        raise RuntimeError("Orchestrator registry not initialized")
    return registry


def _ensure_num_verses(spec: PoemSpec, desired: int | None, default_num: int) -> PoemSpec:
    num = desired or spec.num_verses or default_num
    return spec.model_copy(update={"num_verses": num})


@router.post("/generate", response_model=GeneratePoemResponse)
async def generate_poem(
    payload: GeneratePoemRequest,
    registry: ServiceRegistry = Depends(get_registry),
) -> GeneratePoemResponse:
    settings: Settings = registry.settings

    rag_hits = []
    if payload.use_rag:
        try:
            rag_hits = (
                await registry.rag_client.search(payload.user_query, settings.default_top_k_rag)
            ).hits
        except Exception:
            rag_hits = []

    spec = await registry.yehia_client.build_spec(payload.user_query, rag_hits)
    spec = _ensure_num_verses(spec, payload.desired_num_verses, settings.default_num_verses)

    verses: list[VerseWithScore] = []
    previous_verses: list[str] = []

    for idx in range(1, spec.num_verses + 1):
        verse_text = await registry.shaer_client.generate_bayt(spec, idx, previous_verses)
        scoring = await registry.scoring_client.score_bayt(
            verse_text, spec, previous_verses, use_openai_scoring=payload.use_openai_scoring
        )
        attempts = 1
        while not scoring.passed and attempts < settings.max_retries_per_bayt:
            summary = summarize_feedback(scoring)
            verse_text = await registry.enhancer_client.enhance_bayt(
                spec,
                previous_verses,
                idx,
                verse_text,
                scoring,
                summary,
            )
            scoring = await registry.scoring_client.score_bayt(
                verse_text, spec, previous_verses, use_openai_scoring=payload.use_openai_scoring
            )
            attempts += 1
        previous_verses.append(verse_text)
        verses.append(VerseWithScore(text=verse_text, scoring=scoring))

    return GeneratePoemResponse(
        spec=spec,
        verses=verses,
        rag_hits=[h.poem_id for h in rag_hits] if rag_hits else None,
    )
