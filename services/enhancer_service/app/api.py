from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request

from .clients import ShaerServiceClient
from .config import Settings, get_settings
from .policy import EnhancementPlanner, normalize_candidate_text
from .schemas import EnhanceBaytRequest, EnhanceBaytResponse

router = APIRouter()


def get_planner(request: Request) -> EnhancementPlanner:
    planner = getattr(request.app.state, "enhancement_planner", None)
    if planner is None:
        raise RuntimeError("Enhancement planner has not been initialized")
    return planner


def get_shaer_client(request: Request) -> ShaerServiceClient:
    client = getattr(request.app.state, "shaer_client", None)
    if client is None:
        raise RuntimeError("Shaer client has not been initialized")
    return client


def get_service_settings() -> Settings:
    return get_settings()


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/enhance-bayt", response_model=EnhanceBaytResponse)
async def enhance_bayt(
    payload: EnhanceBaytRequest,
    planner: EnhancementPlanner = Depends(get_planner),
    shaer_client: ShaerServiceClient = Depends(get_shaer_client),
):
    if not payload.request.poem_meter:
        raise HTTPException(status_code=400, detail="poem_meter is required in the request payload")

    plan = planner.build_plan(
        payload.request,
        payload.meter_eval,
        payload.yehia_feedback,
        payload.feedback_summary,
    )

    try:
        candidate = await shaer_client.generate_bayt(plan.request)
    except Exception as exc:  # pragma: no cover - network/HTTP errors
        raise HTTPException(status_code=502, detail=str(exc))

    normalized = normalize_candidate_text(candidate)

    return EnhanceBaytResponse(
        candidate_verse=normalized,
        applied_changes=plan.applied_changes,
        debug={
            "extra_guidance": plan.request.extra_guidance,
            "poem_description": plan.request.poem_description,
        },
    )
