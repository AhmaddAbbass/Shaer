from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from .clients import OrchestratorClient, get_client
from .schemas import ProxyRequest, ProxyResponse

router = APIRouter()


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/proxy", response_model=ProxyResponse)
async def proxy_endpoint(
    request: ProxyRequest,
    orch: OrchestratorClient = Depends(get_client),
) -> ProxyResponse:
    path = request.target or "/poem/generate"
    if not path.startswith("/"):
        path = "/" + path
    try:
        data = await orch.forward_json(path, request.payload)
    except Exception as exc:  # pragma: no cover - pass-through errors
        raise HTTPException(status_code=502, detail=str(exc))
    return ProxyResponse(data=data)
