from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from .config import Settings, get_settings
from .scansion import ScansionService
from .schemas import EvalBaytRequest, EvalBaytResponse, HealthResponse

router = APIRouter()


# Dependency helpers -----------------------------------------------------
def get_scansion_service(settings: Settings = Depends(get_settings)) -> ScansionService:
    if not hasattr(settings, "_scansion_service"):
        settings._scansion_service = ScansionService(settings)  # type: ignore[attr-defined]
    return settings._scansion_service  # type: ignore[attr-defined]


# Routes -----------------------------------------------------------------
@router.get("/health", response_model=HealthResponse)
async def health(scansion: ScansionService = Depends(get_scansion_service)):
    return HealthResponse(status="ok" if scansion.assets_loaded else "degraded", assets_loaded=scansion.assets_loaded)


@router.post("/eval-bayt", response_model=EvalBaytResponse)
async def eval_bayt(
    payload: EvalBaytRequest,
    scansion: ScansionService = Depends(get_scansion_service),
):
    try:
        result = scansion.evaluate_bayt(payload.verse_text, payload.target_meter)
    except RuntimeError as exc:
        # Should not happen now that evaluate_bayt falls back, but keep for safety.
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail="Failed to evaluate bayt") from exc

    return EvalBaytResponse(result=result)
