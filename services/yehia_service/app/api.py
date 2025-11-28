from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from services.common_schemas.schemas import PoemSpec, YehiaFeedback

from .client import YehiaRunpodClient, get_client
from .schemas import BuildSpecRequest, BuildSpecResponse, ChatRequest, ChatResponse, FeedbackRequest, FeedbackResponse

router = APIRouter()


@router.get("/health")
async def health() -> dict[str, str]:
    # Config validation happens when client is built; no external calls here.
    get_client()
    return {"status": "ok"}


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    client: YehiaRunpodClient = Depends(get_client),
) -> ChatResponse:
    try:
        text = await client.chat(request)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    return ChatResponse(text=text)


@router.post("/build-spec", response_model=BuildSpecResponse)
async def build_spec_endpoint(
    request: BuildSpecRequest,
    client: YehiaRunpodClient = Depends(get_client),
) -> BuildSpecResponse:
    try:
        spec: PoemSpec = await client.build_spec(request)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    return BuildSpecResponse(spec=spec)


@router.post("/feedback", response_model=FeedbackResponse)
async def feedback_endpoint(
    request: FeedbackRequest,
    client: YehiaRunpodClient = Depends(get_client),
) -> FeedbackResponse:
    try:
        fb: YehiaFeedback = await client.feedback(request)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    return FeedbackResponse(feedback=fb)
