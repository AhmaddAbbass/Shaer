from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

import httpx

from services.common_schemas.schemas import LLMMessage, PoemSpec, YehiaFeedback

from .config import Settings, get_settings
from .prompt import build_chat_messages, build_feedback_messages, build_spec_messages
from .schemas import BuildSpecRequest, ChatRequest, FeedbackRequest


class YehiaRunpodClient:
    """HTTP client that proxies requests to the Yehia RunPod endpoint."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.logger = logging.getLogger("yehia_service.runpod_client")

    def _to_runpod_messages(self, messages: List[LLMMessage]) -> List[Dict[str, str]]:
        return [{"role": m.role, "content": m.content} for m in messages]

    def _extract_text(self, response_data: Dict[str, Any]) -> str:
        output = response_data.get("output")
        candidates: list[str] = []

        def push(val: Any) -> None:
            if isinstance(val, str) and val.strip():
                candidates.append(val.strip())
            elif isinstance(val, list):
                tokens = [t for t in val if isinstance(t, str)]
                if tokens:
                    joined = "".join(tokens).strip()
                    if joined:
                        candidates.append(joined)

        def walk(val: Any) -> None:
            if isinstance(val, str):
                push(val)
            elif isinstance(val, dict):
                for key in ("text", "generated_text", "output_text", "content", "response", "result"):
                    if key in val:
                        walk(val[key])
                choices = val.get("choices")
                if isinstance(choices, list):
                    for choice in choices:
                        if isinstance(choice, dict):
                            walk(choice.get("message"))
                            walk(choice.get("text"))
                            walk(choice.get("tokens"))
                message = val.get("message")
                if isinstance(message, dict):
                    walk(message.get("content"))
            elif isinstance(val, list):
                push(val)
                for item in val:
                    walk(item)

        walk(output)
        if candidates:
            return candidates[0]
        self.logger.error("Unexpected RunPod response payload: %s", response_data)
        raise RuntimeError("RunPod response did not include generated text.")

    async def _call_runpod(self, messages: List[LLMMessage]) -> str:
        payload = {
            "input": {
                "messages": self._to_runpod_messages(messages),
                "temperature": self.settings.temperature,
                "top_p": self.settings.top_p,
                "max_tokens": self.settings.max_new_tokens,
            }
        }
        url = f"{self.settings.runpod_base_url}/{self.settings.yehia_endpoint_id}/runsync"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.settings.runpod_api_key}",
        }
        async with httpx.AsyncClient(timeout=self.settings.request_timeout) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
        return self._extract_text(data)

    async def chat(self, request: ChatRequest) -> str:
        messages = build_chat_messages(request.messages)
        return await self._call_runpod(messages)

    async def build_spec(self, request: BuildSpecRequest) -> PoemSpec:
        messages = build_spec_messages(request.user_query, request.rag_hits)
        raw_text = await self._call_runpod(messages)
        return self._parse_spec_text(raw_text, fallback_query=request.user_query)

    def _parse_spec_text(self, text: str, fallback_query: str) -> PoemSpec:
        try:
            parsed = json.loads(text)
        except Exception:
            parsed = {}

        def _get_str(key: str, default: str = "") -> str:
            val = parsed.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
            return default

        def _get_int(key: str, default: int) -> int:
            val = parsed.get(key)
            try:
                iv = int(val)
                if iv > 0:
                    return iv
            except Exception:
                pass
            return default

        poem_meter = _get_str("poem_meter", "بحر غير محدد")
        poem_description = _get_str("poem_description", fallback_query.strip() or "قصيدة قصيرة")
        num_verses = _get_int("num_verses", 4)

        return PoemSpec(
            poem_meter=poem_meter,
            poem_description=poem_description,
            num_verses=num_verses,
            poem_title=_get_str("poem_title", None) or None,
            poem_theme=_get_str("poem_theme", None) or None,
            poem_era=_get_str("poem_era", None) or None,
            poet_name=_get_str("poet_name", None) or None,
        )

    async def feedback(self, request: FeedbackRequest) -> YehiaFeedback:
        messages = build_feedback_messages(request.verse_text, request.spec)
        raw_text = await self._call_runpod(messages)
        return self._parse_feedback_text(raw_text)

    def _parse_feedback_text(self, text: str) -> YehiaFeedback:
        try:
            parsed = json.loads(text)
        except Exception:
            parsed = {}

        ok_val = parsed.get("ok")
        ok = bool(ok_val) if ok_val is not None else False

        score_val = parsed.get("score")
        try:
            score = int(score_val) if score_val is not None else None
        except Exception:
            score = None

        feedback = parsed.get("feedback") if isinstance(parsed.get("feedback"), str) else ""
        if not feedback:
            feedback = text.strip()

        return YehiaFeedback(ok=ok, score=score, feedback=feedback)


_CLIENT: YehiaRunpodClient | None = None


def get_client() -> YehiaRunpodClient:
    global _CLIENT
    if _CLIENT is None:
        settings = get_settings()
        _CLIENT = YehiaRunpodClient(settings)
    return _CLIENT
