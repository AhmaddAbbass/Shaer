from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from .aggregator import ScoringAggregator
from .schemas import ScoreBaytRequest, ScoreResponse

router = APIRouter()


def get_aggregator(request: Request) -> ScoringAggregator:
    aggregator = getattr(request.app.state, "scoring_aggregator", None)
    if aggregator is None:
        raise RuntimeError("Scoring aggregator is not initialized")
    return aggregator


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/score", response_model=ScoreResponse)
async def score(
    payload: ScoreBaytRequest,
    aggregator: ScoringAggregator = Depends(get_aggregator),
) -> ScoreResponse:
    return await aggregator.score(payload)
