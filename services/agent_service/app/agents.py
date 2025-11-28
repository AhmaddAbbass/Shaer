from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from services.common_schemas.schemas import (
    LLMMessage,
    PoemSpec,
    RagSearchHit,
)

from .clients import MeterServiceClient, RagServiceClient, ShaerServiceClient, YehiaServiceClient
from .config import Settings
from .schemas import AgentStep, LibraryItem
from .utils import extract_candidate_verses, format_poem_snippet


@dataclass
class PoetryResult:
    spec: PoemSpec
    verses: List[str]
    original_verses: Optional[List[str]] = None
    notes: Optional[str] = None
    library_items: List[LibraryItem] = None
    warnings: List[str] = None

    def __post_init__(self) -> None:
        if self.library_items is None:
            self.library_items = []
        if self.warnings is None:
            self.warnings = []


class TraceRecorder:
    def __init__(self, start_step: int = 1):
        self._next_step = start_step
        self.steps: List[AgentStep] = []

    def add(self, agent: str, tool: str, summary: str) -> None:
        step = AgentStep(step=self._next_step, agent=agent, tool=tool, summary=summary)
        self.steps.append(step)
        self._next_step += 1

    @property
    def current_step(self) -> int:
        return self._next_step


class PoetryAgent:
    def __init__(
        self,
        settings: Settings,
        rag_client: RagServiceClient,
        yehia_client: YehiaServiceClient,
        shaer_client: ShaerServiceClient,
        meter_client: MeterServiceClient,
    ):
        self.settings = settings
        self.rag_client = rag_client
        self.yehia_client = yehia_client
        self.shaer_client = shaer_client
        self.meter_client = meter_client

    async def generate_poem(self, user_query: str, trace: TraceRecorder) -> PoetryResult:
        warnings: List[str] = []
        rag_hits, rag_warning = await self._safe_rag_search(user_query, trace)
        if rag_warning:
            warnings.append(rag_warning)
        spec = await self._build_spec(user_query, rag_hits, trace)
        trace.add("Poetry", "compose_poem", f"توليد قصيدة من {spec.num_verses} أبيات وفق الوصف المطلوب.")
        verses, verse_warnings = await self._compose_poem(spec, trace)
        warnings.extend(verse_warnings)
        library_items = self._hits_to_library_items(rag_hits)
        return PoetryResult(spec=spec, verses=verses, library_items=library_items, warnings=warnings)

    async def fix_poem(self, user_query: str, trace: TraceRecorder) -> PoetryResult:
        original_verses = extract_candidate_verses(user_query)
        if not original_verses:
            raise ValueError("لم يتم العثور على أبيات في طلب المستخدم لإصلاحها.")
        warnings: List[str] = []
        rag_hits, rag_warning = await self._safe_rag_search(user_query, trace)
        if rag_warning:
            warnings.append(rag_warning)
        spec = await self._build_spec(user_query, rag_hits, trace)
        spec = spec.model_copy(update={"num_verses": len(original_verses)})
        trace.add(
            "Poetry",
            "evaluate_poem",
            "تم إرسال الأبيات للتحقق من الوزن والمعنى تمهيداً لإعادة الصياغة عند الحاجة.",
        )
        fixed_verses, verse_warnings = await self._repair_verses(spec, original_verses, trace)
        warnings.extend(verse_warnings)
        library_items = self._hits_to_library_items(rag_hits)
        notes = "تمت المحافظة على الأبيات السليمة وإعادة توليد الأبيات المكسورة." if warnings else None
        return PoetryResult(
            spec=spec,
            verses=fixed_verses,
            original_verses=original_verses,
            notes=notes,
            library_items=library_items,
            warnings=warnings,
        )

    async def _safe_rag_search(self, query: str, trace: TraceRecorder) -> tuple[List[RagSearchHit], Optional[str]]:
        try:
            response = await self.rag_client.search(query, top_k=self.settings.rag_top_k)
            hits = response.hits
            if hits:
                trace.add("Library", "search_poems", "تم استرجاع أمثلة مشابهة لدعم بناء المواصفات.")
            else:
                trace.add("Library", "search_poems", "لم يتم العثور على أمثلة مناسبة، سيتم الاعتماد على وصف المستخدم فقط.")
            warning = None if hits else "لم يتم العثور على قصائد مشابهة، تم الاعتماد على وصف المستخدم فقط."
            return hits, warning
        except Exception:
            trace.add("Library", "search_poems", "تعذر الوصول لخدمة RAG، سيتم استخدام قيم افتراضية.")
            return [], "تعذر الوصول لخدمة الاسترجاع، تم استخدام القيم الافتراضية لإكمال الطلب."

    async def _build_spec(self, user_query: str, hits: List[RagSearchHit], trace: TraceRecorder) -> PoemSpec:
        try:
            spec = await self.yehia_client.build_spec(user_query=user_query, rag_hits=hits)
        except Exception:
            trace.add("Poetry", "build_spec", "تعذر على يحيى بناء المواصفات، سيتم إنشاء مواصفات افتراضية.")
            spec = PoemSpec(
                poem_meter=self.settings.default_meter,
                poem_description=user_query.strip() or "قصيدة قصيرة",
                poem_theme=self.settings.default_theme,
                num_verses=self.settings.default_num_verses,
            )
            return spec

        update_data = {}
        if not spec.poem_meter or "غير" in spec.poem_meter:
            update_data["poem_meter"] = self.settings.default_meter
        if not spec.poem_theme:
            update_data["poem_theme"] = self.settings.default_theme
        if not spec.poem_description.strip():
            update_data["poem_description"] = user_query.strip() or "قصيدة قصيرة"
        if not spec.num_verses or spec.num_verses <= 0:
            update_data["num_verses"] = self.settings.default_num_verses
        if update_data:
            spec = spec.model_copy(update=update_data)
            trace.add("Poetry", "build_spec", "تم تعديل بعض الحقول بقيم افتراضية لضمان جاهزية المواصفات.")
        else:
            trace.add("Poetry", "build_spec", "تم بناء مواصفات القصيدة بالاعتماد على يحيى والأمثلة المسترجعة.")
        return spec

    async def _compose_poem(self, spec: PoemSpec, trace: TraceRecorder) -> tuple[List[str], List[str]]:
        verses: List[str] = []
        warnings: List[str] = []
        for seq in range(1, spec.num_verses + 1):
            verse_text, verse_warnings, regenerated = await self._generate_single_bayt(seq, spec, verses)
            verses.append(verse_text)
            warnings.extend(verse_warnings)
            if regenerated:
                trace.add("Poetry", "enhance_poem", f"إعادة توليد البيت رقم {seq} لتحسين الوزن أو المعنى.")
        trace.add("Poetry", "evaluate_poem", "تم تقييم القصيدة بالكامل بعد التوليد.")
        return verses, warnings

    async def _generate_single_bayt(
        self,
        sequence_number: int,
        spec: PoemSpec,
        previous_verses: List[str],
    ) -> tuple[str, List[str], bool]:
        warnings: List[str] = []
        regenerated = False
        attempt = 0
        verse_text = ""
        while attempt < self.settings.max_bayt_retries:
            attempt += 1
            try:
                verse_text = await self.shaer_client.generate_bayt(spec, sequence_number, previous_verses)
                verse_text = self._clean_verse_text(verse_text)
            except Exception:
                warnings.append("تعذر الاتصال بخدمة شاعر، سيتم استخدام نص بديل بسيط.")
                verse_text = "" if attempt < self.settings.max_bayt_retries else "قصيدة مؤقتة بلا وزن واضح."

            meter_ok = await self._check_meter(verse_text, spec, warnings)
            semantic_ok = await self._check_semantics(verse_text, spec, warnings)
            if meter_ok and semantic_ok and verse_text.strip():
                break
            regenerated = True
        if not verse_text.strip():
            verse_text = "بيت غير مكتمل"
            warnings.append("تم إنشاء بيت افتراضي لعدم توفر استجابة من الخدمات.")
        return verse_text.strip(), warnings, regenerated

    async def _check_meter(self, verse_text: str, spec: PoemSpec, warnings: List[str]) -> bool:
        try:
            result = await self.meter_client.eval_bayt(verse_text, spec.poem_meter)
            return bool(result.on_meter)
        except Exception:
            warnings.append("تعذر فحص البحر، تم اعتبار البيت مقبولاً مع استخدام البحر الافتراضي.")
            return True

    async def _check_semantics(self, verse_text: str, spec: PoemSpec, warnings: List[str]) -> bool:
        try:
            feedback = await self.yehia_client.feedback(verse_text, spec)
            return bool(feedback.ok)
        except Exception:
            warnings.append("تعذر الحصول على تغذية راجعة من يحيى، تم قبول البيت مؤقتاً.")
            return True

    async def _repair_verses(
        self,
        spec: PoemSpec,
        original_verses: List[str],
        trace: TraceRecorder,
    ) -> tuple[List[str], List[str]]:
        fixed: List[str] = []
        warnings: List[str] = []
        for idx, verse in enumerate(original_verses, start=1):
            meter_ok = await self._check_meter(verse, spec, warnings)
            semantic_ok = await self._check_semantics(verse, spec, warnings)
            if meter_ok and semantic_ok:
                fixed.append(verse)
                continue
            trace.add("Poetry", "compose_bayt", f"توليد بديل للبيت رقم {idx}.")
            new_verse, verse_warnings, _ = await self._generate_single_bayt(idx, spec, fixed)
            fixed.append(new_verse)
            warnings.extend(verse_warnings)
        return fixed, warnings

    def _hits_to_library_items(self, hits: List[RagSearchHit]) -> List[LibraryItem]:
        items: List[LibraryItem] = []
        for hit in hits[:3]:
            title = hit.poem_title or f"قصيدة {hit.poem_id}"
            snippet = hit.poem_description[:180]
            items.append(
                LibraryItem(
                    poem_id=str(hit.poem_id),
                    title=title,
                    poet_name=hit.poet_name,
                    poem_meter=hit.poem_meter,
                    poem_era=hit.poem_era,
                    poem_theme=hit.poem_theme,
                    snippet=snippet,
                )
            )
        return items

    def _clean_verse_text(self, raw_text: str) -> str:
        """
        Keep only the main generated bayt and drop any prompt echoes or system markers
        (e.g. `[INST]`, `<<SYS>>`, user instructions) that Shaer might emit when it
        returns the full conversation transcript.
        """
        text = raw_text.replace("\r", "\n")
        for marker in ("[INST]", "<<SYS>>", "<</SYS>>", "المطلوب منك"):
            marker_idx = text.find(marker)
            if marker_idx != -1:
                text = text[:marker_idx]
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return lines[0] if lines else raw_text.strip()


class LibraryAgent:
    def __init__(self, rag_client: RagServiceClient):
        self.rag_client = rag_client

    async def search_poems(self, query: str, trace: TraceRecorder, top_k: int) -> tuple[str, List[LibraryItem]]:
        trace.add("Library", "search_poems", "جاري البحث في المكتبة عن قصائد مشابهة.")
        try:
            response = await self.rag_client.search(query, top_k=top_k)
        except Exception:
            trace.add("Library", "search_poems", "تعذر التواصل مع خدمة المكتبة.")
            return ("تعذر البحث في المكتبة حالياً.", [])

        hits = sorted(
            response.hits,
            key=lambda h: (h.poem_era or "", h.poet_name or "", h.poem_title or ""),
        )
        items: List[LibraryItem] = []
        reply_parts: List[str] = []
        for hit in hits:
            poem = await self.rag_client.get_poem(str(hit.poem_id))
            snippet = format_poem_snippet(poem.verses if poem else []) if poem else hit.poem_description[:200]
            title = hit.poem_title or (poem.poem_title if poem else f"قصيدة {hit.poem_id}")
            line = f"{hit.poem_era or 'عصر غير محدد'} – {hit.poet_name or 'شاعر غير معروف'} – {title}\n{snippet}".strip()
            reply_parts.append(line)
            items.append(
                LibraryItem(
                    poem_id=str(hit.poem_id),
                    title=title,
                    poet_name=hit.poet_name,
                    poem_meter=hit.poem_meter,
                    poem_era=hit.poem_era,
                    poem_theme=hit.poem_theme,
                    snippet=snippet,
                )
            )
        if not reply_parts:
            reply_text = "لم أعثر على قصائد مطابقة، حاول تضييق طلبك أو تحديد شاعر أو عصر."
        else:
            reply_text = "\n\n".join(reply_parts)
        return reply_text, items


class ExplainAgent:
    def __init__(self, settings: Settings, yehia_client: YehiaServiceClient):
        self.settings = settings
        self.yehia_client = yehia_client

    async def explain(self, user_message: str, trace: TraceRecorder) -> str:
        trace.add("Explain", "explain_query", "تم تحويل السؤال ليحيى لشرح المفهوم المطلوب.")
        messages = [
            LLMMessage(role="system", content=self.settings.explanation_system_prompt),
            LLMMessage(role="user", content=user_message),
        ]
        try:
            return await self.yehia_client.chat(messages)
        except Exception:
            return "تعذر الحصول على شرح من يحيى حالياً."
