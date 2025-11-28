from __future__ import annotations

from typing import List, Optional

from fastapi import HTTPException

from services.common_schemas.schemas import PoemSpec

from .agents import ExplainAgent, LibraryAgent, PoetryAgent, PoetryResult, TraceRecorder
from .clients import MeterServiceClient, RagServiceClient, ShaerServiceClient, YehiaServiceClient
from .config import Settings
from .schemas import ChatMessage, ChatRequest, ChatResponse, LibraryItem, PoemVersion
from .tool_selector import ToolSelector, ToolSelectionResult

ALLOWED_MODES = {"generate", "fix", "search", "explain"}


class OrchestratorService:
    def __init__(
        self,
        settings: Settings,
        yehia_client: YehiaServiceClient,
        shaer_client: ShaerServiceClient,
        rag_client: RagServiceClient,
        meter_client: MeterServiceClient,
    ):
        self.settings = settings
        self.yehia_client = yehia_client
        self.shaer_client = shaer_client
        self.rag_client = rag_client
        self.meter_client = meter_client

        self.poetry_agent = PoetryAgent(settings, rag_client, yehia_client, shaer_client, meter_client)
        self.library_agent = LibraryAgent(rag_client)
        self.explain_agent = ExplainAgent(settings, yehia_client)
        self.tool_selector = (
            ToolSelector(settings.openai_api_key, settings.openai_model, settings.tool_selection_system_prompt)
            if settings.openai_api_key
            else None
        )

    async def handle_chat(self, request: ChatRequest) -> ChatResponse:
        latest_user = self._latest_user_message(request.messages)
        if not latest_user:
            raise HTTPException(status_code=400, detail="يجب أن تحتوي المحادثة على رسالة مستخدم واحدة على الأقل.")

        trace = TraceRecorder()
        selection, selection_error = await self._select_mode_via_llm(latest_user)
        if selection:
            mode = selection.mode
            reason = f" – {selection.reason}" if selection.reason else ""
            trace.add("Orchestrator", "gpt_mode_selector", f"GPT-4o اختار الأداة: {mode}{reason}")
        else:
            mode = self._infer_mode(request.mode, latest_user)
            if selection_error:
                trace.add("Orchestrator", "gpt_mode_selector", selection_error)
        trace.add("Orchestrator", "parse_intent", f"تم تحديد نية المستخدم: {mode}.")

        warnings: List[str] = []
        poem_spec: Optional[PoemSpec] = None
        poem_version: Optional[PoemVersion] = None
        library_items: List[LibraryItem] = []

        if mode == "generate":
            result = await self.poetry_agent.generate_poem(latest_user.content, trace)
            poem_spec = result.spec
            poem_version = PoemVersion(spec=result.spec, verses=result.verses)
            library_items = result.library_items
            warnings.extend(result.warnings)
            reply = self._format_poem_reply(result.verses, title=result.spec.poem_title)
        elif mode == "fix":
            try:
                result = await self.poetry_agent.fix_poem(latest_user.content, trace)
            except ValueError as exc:
                return ChatResponse(
                    reply=str(exc),
                    mode=mode,
                    agent_trace=trace.steps,
                    warnings=[str(exc)],
                )
            poem_spec = result.spec
            poem_version = PoemVersion(
                spec=result.spec,
                verses=result.verses,
                original_verses=result.original_verses,
                notes=result.notes,
            )
            library_items = result.library_items
            warnings.extend(result.warnings)
            reply = self._format_poem_reply(result.verses, title=result.spec.poem_title)
        elif mode == "search":
            reply, library_items = await self.library_agent.search_poems(latest_user.content, trace, self.settings.rag_top_k)
        elif mode == "explain":
            reply = await self.explain_agent.explain(latest_user.content, trace)
        else:
            # fallback to generate
            result = await self.poetry_agent.generate_poem(latest_user.content, trace)
            poem_spec = result.spec
            poem_version = PoemVersion(spec=result.spec, verses=result.verses)
            library_items = result.library_items
            warnings.extend(result.warnings)
            reply = self._format_poem_reply(result.verses, title=result.spec.poem_title)

        return ChatResponse(
            reply=reply,
            mode=mode,
            poem_spec=poem_spec,
            poem_version=poem_version,
            agent_trace=trace.steps,
            library_context=library_items,
            warnings=warnings,
        )

    def _latest_user_message(self, messages: List[ChatMessage]) -> Optional[ChatMessage]:
        for msg in reversed(messages):
            if msg.role == "user" and msg.content.strip():
                return msg
        return None

    def _infer_mode(self, mode_hint: Optional[str], message: ChatMessage) -> str:
        normalized = (mode_hint or "").strip().lower()
        if normalized in ALLOWED_MODES:
            return normalized

        text = message.content
        lowered = text.lower()
        if any(keyword in lowered for keyword in ["اصلح", "عد", "صحح"]):
            return "fix"
        if any(keyword in lowered for keyword in ["ابحث", "أرني", "ارني", "اعرض", "أريد قصيدة"]):
            return "search"
        if any(keyword in lowered for keyword in ["اشرح", "ما هو", "فسر"]):
            return "explain"
        return "generate"

    def _format_poem_reply(self, verses: List[str], title: Optional[str]) -> str:
        lines = []
        if title:
            lines.append(f"{title}\n")
        lines.extend(verses)
        return "\n".join(lines)

    async def aclose(self) -> None:
        await self.yehia_client.close()
        await self.shaer_client.close()
        await self.rag_client.close()
        await self.meter_client.close()

    async def _select_mode_via_llm(
        self, message: ChatMessage
    ) -> tuple[Optional[ToolSelectionResult], Optional[str]]:
        if not self.tool_selector or not message.content.strip():
            return None, None
        try:
            result = await self.tool_selector.select_mode(message.content, ALLOWED_MODES)
        except Exception:
            return None, "تعذر الاتصال بـ GPT-4o لاختيار الأداة، سيتم استخدام الاستدلال المحلي."
        if result is None:
            return None, "لم نحصل على استجابة مفهومة من GPT-4o، سيتم استخدام الاستدلال المحلي."
        return result, None
