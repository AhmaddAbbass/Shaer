from __future__ import annotations

import json
from typing import List

from services.common_schemas.schemas import LLMMessage, PoemSpec, RagSearchHit, YehiaFeedback


SPEC_SYSTEM_PROMPT = (
    "أنت يحيى، شاعر وناقد عربي. استخرج مواصفات قصيدة جديدة من طلب المستخدم "
    "وأعدها في كائن JSON بوضوح دون شرح إضافي. الحقول المطلوبة: "
    "{poem_meter, poem_description, num_verses, poem_theme?, poem_era?, poet_name?, poem_title?}. "
    "احرص أن تكون poem_meter نصاً عربياً واضحاً، poem_description جملة/فقرة قصيرة، "
    "num_verses > 0."
)

FEEDBACK_SYSTEM_PROMPT = (
    "أنت يحيى، ناقد شعري. قيم البيت مقابل المواصفات وأعد كائناً JSON فقط "
    "بالشكل {ok: bool, score: int?, feedback: str}. لا تضف شرحاً خارج JSON."
)

ASPECT_SYSTEM_PROMPTS = {
    "meaning": (
        "أنت يحيى، ناقد شعري يركز على المعنى. قيّم مدى التزام البيت بالوصف المطلوب، ووضوح الفكرة، وعمق الرسالة. "
        "أعد كائناً JSON بالشكل {ok: bool, score: int?, feedback: str} دون أي شرح إضافي."
    ),
    "cohesion": (
        "أنت يحيى، ناقد شعري يركز على الترابط. قيّم انسجام هذا البيت مع الأبيات السابقة والوصف العام، واذكر إن كان هناك انقطاع أو تناقض. "
        "أعد JSON فقط بالشكل {ok, score, feedback}."
    ),
    "fluency": (
        "أنت يحيى، ناقد لغوي. قيّم فصاحة البيت وسلامة تراكيبه النحوية والصوتية، وأعد JSON فقط بالشكل {ok, score, feedback}."
    ),
    "poeticness": (
        "أنت يحيى، ناقد شعري يركز على الشعرية والصور الجمالية. قيّم قوة الصور والاستعارات والإيقاع المعنوي، "
        "ثم أعد JSON فقط بالشكل {ok, score, feedback}."
    ),
}


def build_spec_messages(user_query: str, rag_hits: List[RagSearchHit]) -> List[LLMMessage]:
    rag_block = ""
    if rag_hits:
        parts = []
        for hit in rag_hits[:5]:
            parts.append(
                f"- عنوان غير معروف | شاعر: {hit.poet_name or 'غير محدد'} | بحر: {hit.poem_meter or 'غير محدد'} | وصف: {hit.poem_description}"
            )
        rag_block = "أمثلة مسترجعة:\n" + "\n".join(parts)

    user_content = f"طلب المستخدم:\n{user_query}\n\n{rag_block}\n\nأعد كائناً JSON فقط."
    return [
        LLMMessage(role="system", content=SPEC_SYSTEM_PROMPT),
        LLMMessage(role="user", content=user_content),
    ]


def _format_previous(previous_verses: list[str]) -> str:
    lines = [v.strip() for v in previous_verses if v and v.strip()]
    if not lines:
        return "لا توجد أبيات سابقة."
    return "\n".join(f"{idx}. {line}" for idx, line in enumerate(lines, start=1))


def build_feedback_messages(verse_text: str, spec: PoemSpec, aspect: str | None = None, previous_verses: list[str] | None = None) -> List[LLMMessage]:
    spec_json = json.dumps(spec.model_dump(), ensure_ascii=False, indent=2)
    previous_block = _format_previous(previous_verses or [])
    user_content = (
        f"المواصفات:\n{spec_json}\n\nالبيت:\n{verse_text}\n\n"
        f"الأبيات السابقة (إن وجدت):\n{previous_block}\n\n"
        "أعد كائناً JSON فقط بالشكل {ok, score, feedback}."
    )
    system_prompt = ASPECT_SYSTEM_PROMPTS.get((aspect or "").lower(), FEEDBACK_SYSTEM_PROMPT)
    return [
        LLMMessage(role="system", content=system_prompt),
        LLMMessage(role="user", content=user_content),
    ]


def build_chat_messages(messages: List[LLMMessage]) -> List[LLMMessage]:
    return messages


__all__ = [
    "build_spec_messages",
    "build_feedback_messages",
    "build_chat_messages",
    "SPEC_SYSTEM_PROMPT",
    "FEEDBACK_SYSTEM_PROMPT",
    "ASPECT_SYSTEM_PROMPTS",
]
