from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request

from .evaluator import AspectEvaluator
from .prompts import ASPECTS
from .schemas import (
    EvaluateAspectRequest,
    EvaluateAspectResponse,
    EvaluateBaytRequest,
    EvaluateBaytResponse,
)

router = APIRouter()


def get_evaluator(request: Request) -> AspectEvaluator:
    evaluator = getattr(request.app.state, "aspect_evaluator", None)
    if evaluator is None:
        raise RuntimeError("Aspect evaluator is not initialized")
    return evaluator


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/evaluate-bayt", response_model=EvaluateBaytResponse)
async def evaluate_bayt(
    payload: EvaluateBaytRequest,
    evaluator: AspectEvaluator = Depends(get_evaluator),
) -> EvaluateBaytResponse:
    results = await evaluator.evaluate_all(
        verse_text=payload.verse_text,
        spec=payload.spec,
        previous_verses=payload.previous_verses,
        use_openai=payload.use_openai_scoring,
    )
    return EvaluateBaytResponse(**results)


@router.post("/evaluate-aspect", response_model=EvaluateAspectResponse)
async def evaluate_aspect(
    payload: EvaluateAspectRequest,
    evaluator: AspectEvaluator = Depends(get_evaluator),
) -> EvaluateAspectResponse:
    aspect = payload.aspect
    if aspect not in ASPECTS:
        raise HTTPException(status_code=400, detail="Unsupported aspect")
    result = await evaluator.evaluate_one(
        aspect=aspect,
        verse_text=payload.verse_text,
        spec=payload.spec,
        previous_verses=payload.previous_verses,
        use_openai=payload.use_openai_scoring,
    )
    return EvaluateAspectResponse(aspect=aspect, evaluation=result)
