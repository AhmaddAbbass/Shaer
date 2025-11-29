from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from ..context import ServiceRegistry
from ..schemas import LibrarySearchRequest, LibrarySearchResponse

router = APIRouter(prefix="/library", tags=["library"])


def get_registry(request: Request) -> ServiceRegistry:
    registry = getattr(request.app.state, "registry", None)
    if registry is None:
        raise RuntimeError("Orchestrator registry not initialized")
    return registry


@router.post("/search", response_model=LibrarySearchResponse)
async def library_search(
    payload: LibrarySearchRequest,
    registry: ServiceRegistry = Depends(get_registry),
) -> LibrarySearchResponse:
    results = await registry.rag_client.search(payload.query_text, payload.top_k)
    return LibrarySearchResponse(results=results)
