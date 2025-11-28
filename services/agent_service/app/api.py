from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from .orchestrator import OrchestratorService
from .schemas import ChatRequest, ChatResponse

router = APIRouter()


def get_orchestrator(request: Request) -> OrchestratorService:
    orchestrator = getattr(request.app.state, "orchestrator", None)
    if orchestrator is None:
        raise RuntimeError("Orchestrator service is not initialized")
    return orchestrator


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(
    payload: ChatRequest,
    orchestrator: OrchestratorService = Depends(get_orchestrator),
) -> ChatResponse:
    return await orchestrator.handle_chat(payload)
