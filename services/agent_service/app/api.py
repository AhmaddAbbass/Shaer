from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from .clients import OrchestratorClient, get_client
from .schemas import (
    ChatRequest,
    ChatResponse,
    ProxyRequest,
    ProxyResponse,
)
from .logging import get_logger

router = APIRouter()
logger = get_logger("agent_service.api")


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


@router.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    orch: OrchestratorClient = Depends(get_client),
) -> ChatResponse:
    logger.info("UI chat: mode=%s messages=%d", request.mode, len(request.messages))
    # Find last user message
    user_text = ""
    for msg in reversed(request.messages):
        if msg.role == "user" and msg.content.strip():
            user_text = msg.content.strip()
            break
    if not user_text:
        logger.warning("UI chat: rejected request with no user message")
        raise HTTPException(status_code=400, detail="No user message provided")

    mode = (request.mode or "generate").lower()
    try:
        if mode == "search":
            data = await orch.forward_json("/library/search", {"query_text": user_text, "top_k": 5})
            library = data.get("results", {}).get("hits") if isinstance(data, dict) else None
            logger.info("UI chat search: hits=%d", len(library or []))
            return ChatResponse(
                reply="تم العثور على نتائج.",
                mode="search",
                library_context=library or [],
            )

        if mode == "fix":
            data = await orch.forward_json(
                "/bayt/fix",
                {
                    "verse_text": user_text,
                    "previous_verses": [],
                    "use_openai_scoring": True,
                },
        )
        verse_text = data.get("verse_text", "") if isinstance(data, dict) else ""
        poem_version = {"spec": data.get("spec"), "verses": [verse_text]} if isinstance(data, dict) else None
        agent_trace = data.get("agent_trace") if isinstance(data, dict) else None
        logger.info("UI chat fix: verse_len=%d", len(verse_text))
        return ChatResponse(
            reply=verse_text or "تم إصلاح البيت.",
            mode="fix",
            poem_version=poem_version,
            agent_trace=agent_trace,
        )

        # Default: generate
        data = await orch.forward_json(
            "/poem/generate",
            {
                "user_query": user_text,
                "desired_num_verses": None,
                "use_rag": True,
                "use_openai_scoring": True,
            },
        )
        spec = data.get("spec") if isinstance(data, dict) else None
        verses = data.get("verses") if isinstance(data, dict) else []
        verse_texts = [v.get("text", "") for v in verses if isinstance(v, dict)] if verses else []
        reply = "\n".join(verse_texts) if verse_texts else "تم توليد قصيدة."
        agent_trace = data.get("agent_trace") if isinstance(data, dict) else None
        logger.info(
            "UI chat generate: verses=%d spec=%s",
            len(verse_texts),
            "yes" if spec else "no",
        )
        return ChatResponse(
            reply=reply,
            mode="generate",
            poem_spec=spec,
            poem_version={"spec": spec, "verses": verse_texts} if spec or verse_texts else None,
            agent_trace=agent_trace,
        )
    except Exception as exc:
        logger.exception("UI chat failed: %s", exc)
        raise HTTPException(status_code=502, detail=str(exc))
