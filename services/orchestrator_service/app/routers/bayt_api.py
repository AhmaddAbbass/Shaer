from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from services.common_schemas.schemas import PoemSpec

from ..context import ServiceRegistry
from ..schemas import FixBaytRequest, FixBaytResponse, ScoreBaytRequest, ScoreBaytResponse
from ..settings import Settings
from ..utils.feedback import summarize_feedback
from ..utils.normalization import normalize_single_bayt

router = APIRouter(prefix="/bayt", tags=["bayt"])


def get_registry(request: Request) -> ServiceRegistry:
    registry = getattr(request.app.state, "registry", None)
    if registry is None:
        raise RuntimeError("Orchestrator registry not initialized")
    return registry


def _spec_from_request(request: FixBaytRequest | ScoreBaytRequest, settings: Settings) -> PoemSpec:
    if request.spec:
        return request.spec
    description = f"إصلاح البيت التالي: {request.verse_text[:80]}"
    return PoemSpec(poem_meter="بحر غير محدد", poem_description=description, num_verses=1)


@router.post("/fix", response_model=FixBaytResponse)
async def fix_bayt(
    payload: FixBaytRequest,
    registry: ServiceRegistry = Depends(get_registry),
) -> FixBaytResponse:
    settings = registry.settings
    verse_text = normalize_single_bayt(payload.verse_text)
    spec = _spec_from_request(payload, settings)
    previous = payload.previous_verses

    scoring = await registry.scoring_client.score_bayt(
        verse_text, spec, previous, use_openai_scoring=payload.use_openai_scoring
    )
    attempts = 0
    while not scoring.passed and attempts < settings.max_retries_per_bayt:
        summary = summarize_feedback(scoring)
        verse_text = await registry.enhancer_client.enhance_bayt(
            spec,
            previous,
            1,
            verse_text,
            scoring,
            summary,
        )
        scoring = await registry.scoring_client.score_bayt(
            verse_text, spec, previous, use_openai_scoring=payload.use_openai_scoring
        )
        attempts += 1
    return FixBaytResponse(verse_text=verse_text, scoring=scoring)


@router.post("/score", response_model=ScoreBaytResponse)
async def score_bayt(
    payload: ScoreBaytRequest,
    registry: ServiceRegistry = Depends(get_registry),
) -> ScoreBaytResponse:
    spec = _spec_from_request(payload, registry.settings)
    verse_text = normalize_single_bayt(payload.verse_text)
    scoring = await registry.scoring_client.score_bayt(
        verse_text, spec, payload.previous_verses, use_openai_scoring=payload.use_openai_scoring
    )
    return ScoreBaytResponse(scoring=scoring)
