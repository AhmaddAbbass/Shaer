from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from .ashaar import AshaarMeterService
from .config import Settings, get_settings
from .schemas import EvalAshaarRequest, EvalAshaarResponse, HealthResponse

router = APIRouter()


def get_ashaar_service(settings: Settings = Depends(get_settings)) -> AshaarMeterService:
    if not hasattr(settings, "_ashaar_service"):
        settings._ashaar_service = AshaarMeterService(settings)  # type: ignore[attr-defined]
    return settings._ashaar_service  # type: ignore[attr-defined]


@router.get("/health", response_model=HealthResponse)
async def health(service: AshaarMeterService = Depends(get_ashaar_service)):
    return HealthResponse(status="ok" if service.assets_loaded else "degraded", assets_loaded=service.assets_loaded)


@router.post("/ashaar-score", response_model=EvalAshaarResponse)
async def ashaar_score(
    payload: EvalAshaarRequest,
    service: AshaarMeterService = Depends(get_ashaar_service),
) -> EvalAshaarResponse:
    try:
        result = service.evaluate_bayt(payload.verse_text)
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail="Failed to compute Ashaar score") from exc

    return EvalAshaarResponse(result=result)
